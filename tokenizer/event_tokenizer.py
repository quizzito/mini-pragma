"""
Step 1 of the tokenizer: the KEY vocabulary.

Every field name that can appear in an event or profile (e.g. "amount",
"type", "channel") gets mapped to a single integer token. This mirrors
§2.2 of the PRAGMA paper: "we tokenise all semantic types (keys) as
single tokens... a vocabulary of ~60 tokens."

We're starting with just this piece — no values, no time yet — so we can
confirm the key-mapping idea works before building the harder parts.
"""
import pandas as pd

# Every field name we might see across all 5 event types + profile state.
# Order doesn't matter, but each key must appear exactly once.

ALL_KEYS = [
    # special tokens (must come first, so their ids are stable/predictable)
    "[PAD]", "[USR]",
    # shared / structural
    "type", "created",
    # card_payment / topup / trading (transaction-like)
    "direction", "amount", "currency", "fee", "description",
    # app_event
    "view",
    # communication
    "channel", "product", "interact",
    # trading
    "symbol", "price", "order_type",
    # profile state
    "user_id", "region", "plan", "tenure_months", "balance", "age_bracket",
]

# Build key -> integer id, and the reverse mapping (useful for debugging later)
KEY_TO_ID = {key: idx for idx, key in enumerate(ALL_KEYS)}
ID_TO_KEY = {idx: key for key, idx in KEY_TO_ID.items()}


def tokenize_key(key: str) -> int:
    """Map a single field name to its integer token id."""
    if key not in KEY_TO_ID:
        raise KeyError(f"Unknown key '{key}' — add it to ALL_KEYS in event_tokenizer.py")
    return KEY_TO_ID[key]


"""
Step 2 of the tokenizer: CATEGORICAL values.

Unlike keys (one shared vocabulary), each categorical field has its own
small vocabulary — e.g. "direction" only ever contains "in"/"out", while
"region" contains "uk"/"us"/"eu"/"au"/"sg". So we build one vocab per field.

Starting with just ONE field (direction) to prove the idea before scaling
to all categorical fields.
"""

CATEGORICAL_VOCABS = {
    "type": {"card_payment": 0, "topup": 1, "app_event": 2, "communication": 3, "trading": 4},
    "direction": {"in": 0, "out": 1, "buy": 2, "sell": 3},
    "currency": {"gbp": 0, "gbx": 1},
    "description": {"groceries": 0, "restaurant": 1, "transport": 2, "shopping": 3, "subscription": 4},
    "view": {"home": 0, "p2p_amount": 1, "confirm_p2p_dialog": 2, "card_settings": 3, "statements": 4, "support_chat": 5},
    "channel": {"email": 0, "push": 1, "sms": 2},
    "product": {"credit_card": 0, "savings": 1, "stocks_shares_isa": 2, "current_account": 3},
    "interact": {"sent": 0, "opened": 1, "interacted": 2},
    "symbol": {"swda": 0, "vwrl": 1, "aapl": 2, "tsla": 3, "btc": 4},
    "order_type": {"market": 0, "limit": 1},
    "region": {"uk": 0, "us": 1, "eu": 2, "au": 3, "sg": 4},
    "plan": {"standard": 0, "plus": 1, "premium": 2, "metal": 3, "ultra": 4},
    "age_bracket": {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4},
}

# Special value-side token, used for padding value positions. Not tied to
# any particular field's vocabulary, since padding isn't a real value.
PAD_VALUE_ID = -1

def tokenize_categorical(field: str, value: str) -> int:
    """Map a categorical value to its integer token, within its field's vocab."""
    if field not in CATEGORICAL_VOCABS:
        raise KeyError(f"No vocabulary defined for categorical field '{field}'")
    field_vocab = CATEGORICAL_VOCABS[field]
    if value not in field_vocab:
        raise KeyError(f"Unknown value '{value}' for field '{field}' — not in its vocabulary")
    return field_vocab[value]

"""
Step 3 of the tokenizer: NUMERICAL values, via percentile bucketing.

Unlike categorical fields (fixed small vocab), a numerical field like "amount"
can take any value. Instead of tokenizing the raw number, we bucket it by
where it falls in the overall distribution — e.g. "this amount is bigger than
73% of all amounts we've seen" -> bucket 7 (of 10). This preserves relative
magnitude while giving us a small, fixed vocabulary (one token per bucket).

Step 3a: compute the bucket boundaries from real data (this only needs to
happen ONCE, using a large sample — like fitting a scaler in scikit-learn).
"""
import numpy as np


def compute_percentile_boundaries(values: list[float], num_buckets: int = 10) -> list[float]:
    """
    Given a list of real observed values, compute the boundaries that split
    them into `num_buckets` equal-sized groups (percentile bins).
    Returns num_buckets - 1 boundary values.
    """
    percentiles = np.linspace(0, 100, num_buckets + 1)[1:-1]  # skip 0th and 100th
    boundaries = np.percentile(values, percentiles)
    return boundaries.tolist()

def bucket_value(value: float, boundaries: list[float]) -> int:
    """
    Given a value and precomputed boundaries, return which bucket it falls
    into (0 to num_buckets-1). Uses np.searchsorted, which finds how many
    boundaries a value is greater than — that count IS the bucket index.
    """
    return int(np.searchsorted(boundaries, value))

"""
Step 4 of the tokenizer: TIME encoding.

Step 4a: elapsed time since the previous event, log-transformed.
Raw seconds-since-last-event can range from 1 second to years — a huge
range that would break normal positional embeddings. The paper's fix
(§2.2): 8 * ln(1 + t/8) compresses big gaps while keeping small, recent
gaps precise. This is NOT bucketing (like amounts) — it's a smooth
transform that feeds directly into the model as a continuous number.
"""
import math


def encode_elapsed_time(seconds_since_last_event: float) -> float:
    """Apply the paper's soft-log transform to compress large time gaps."""
    return 8 * math.log(1 + seconds_since_last_event / 8)

"""
Step 4b: cyclical calendar features (hour of day, day of week, day of month).

Raw numbers like hour=23 and hour=0 look far apart to a model, even though
they're actually adjacent (11pm is right next to midnight). Sine/cosine
encoding fixes this by placing each value on a circle, so the "wrap-around"
point connects smoothly instead of jumping.
"""


def encode_cyclical(value: int, period: int) -> tuple[float, float]:
    """
    Encode a cyclical value (e.g. hour 0-23, period=24) as an (x, y) point
    on a circle, using sine and cosine. Values near the wrap-around point
    (e.g. hour 23 and hour 0) end up close together in (x, y) space.
    """
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)

"""
Step 5: tokenize a WHOLE event, combining keys, values (categorical/
numerical), and time — this ties together everything we've built so far.
"""

# Which fields are numerical vs categorical, so we know how to treat each one.
NUMERICAL_FIELDS = {"amount", "fee", "price", "balance", "tenure_months"}
CATEGORICAL_FIELDS = set(CATEGORICAL_VOCABS.keys())


def tokenize_event(event: dict, numerical_boundaries: dict, seconds_since_last_event: float) -> dict:
    """
    Tokenize one event dict into key-value tokens PLUS time encodings.
    `seconds_since_last_event` must be computed by the caller (it depends on
    the *previous* event in a user's history, which this function doesn't
    have access to on its own).
    """
    tokens = []
    for field, value in event.items():
        if field in ("created", "user_id"):
            continue  # created -> handled separately via time encoding;
                      # user_id -> a row identifier, not a semantic field to tokenize
        if pd.isna(value):
            continue  # this field doesn't apply to this event's type (e.g. "view"
                      # is NaN for a card_payment row) -- nothing to tokenize

        key_token = tokenize_key(field)

        if field in CATEGORICAL_FIELDS:
            value_token = tokenize_categorical(field, value)
        elif field in NUMERICAL_FIELDS:
            if field not in numerical_boundaries:
                raise KeyError(f"No boundaries computed yet for numerical field '{field}'")
            value_token = bucket_value(value, numerical_boundaries[field])
        else:
            raise KeyError(f"Field '{field}' is neither categorical nor numerical — add it to one of the sets above")

        tokens.append((key_token, value_token))

    # Time encodings, computed from the event's own timestamp
    created = event["created"]
    elapsed = encode_elapsed_time(seconds_since_last_event)
    hour_x, hour_y = encode_cyclical(created.hour, period=24)
    dow_x, dow_y = encode_cyclical(created.weekday(), period=7)

    return {
        "key_value_tokens": tokens,
        "elapsed_time": elapsed,
        "hour_sin_cos": (hour_x, hour_y),
        "day_of_week_sin_cos": (dow_x, dow_y),
    }

"""
Step 6: tokenize PROFILE STATE — reuses the same key/categorical/numerical
machinery as events, since the paper treats profile state as "an event-like
format" (§2.1.2). No time-since-last-event here (it's a snapshot, not a
sequence), but the paper does encode "time since life-long milestones" for
profile state -- we're deferring that as a stretch goal, noted in the log.
"""

LABEL_FIELDS = {"credit_default", "fraud", "engagement"}  # prediction targets, never tokenized as input


def tokenize_profile(profile: dict, numerical_boundaries: dict) -> list[tuple[int, int]]:
    """
    Tokenize one user's profile state into (key_token, value_token) pairs.
    Skips user_id (identifier) and the 3 label fields (prediction targets).
    """
    tokens = []
    for field, value in profile.items():
        if field == "user_id" or field in LABEL_FIELDS:
            continue

        key_token = tokenize_key(field)

        if field in CATEGORICAL_FIELDS:
            value_token = tokenize_categorical(field, value)
        elif field in NUMERICAL_FIELDS:
            if field not in numerical_boundaries:
                raise KeyError(f"No boundaries computed yet for numerical field '{field}'")
            value_token = bucket_value(value, numerical_boundaries[field])
        else:
            raise KeyError(f"Field '{field}' is neither categorical nor numerical — add it to one of the sets above")

        tokens.append((key_token, value_token))

    return tokens

"""
Step 7: tokenize a user's FULL HISTORY — profile + all events, combined
into one ordered sequence. This is the actual input shape a model will
eventually consume: [profile_tokens, event_1_tokens, event_2_tokens, ...]
"""


def tokenize_user_history(
    profile: dict,
    events: list[dict],
    numerical_boundaries: dict,
) -> list[dict]:
    """
    Tokenize one user's entire history: profile first, then events in
    chronological order. Returns a list where each item is one "unit"
    (profile or event) with its own key-value tokens (+ time info for events).
    """
    sequence = []

    # Profile goes first, no time info (it's a snapshot, not a timed event)
    profile_tokens = tokenize_profile(profile, numerical_boundaries)
    sequence.append({"kind": "profile", "key_value_tokens": profile_tokens})

    # Events, in order, each with elapsed time since the PREVIOUS event
    previous_time = None
    for event in events:
        if previous_time is None:
            seconds_since_last = 0
        else:
            seconds_since_last = (event["created"] - previous_time).total_seconds()

        event_tokens = tokenize_event(event, numerical_boundaries, seconds_since_last)
        event_tokens["kind"] = "event"
        sequence.append(event_tokens)

        previous_time = event["created"]

    return sequence

"""
Step 8: FLATTEN a user's history into one single sequence of (key, value)
pairs, prepend a [USR] token, and PAD to a fixed length. This is the input
shape the flattened Transformer (v1) actually consumes.
"""

USR_KEY_ID = tokenize_key("[USR]")
PAD_KEY_ID = tokenize_key("[PAD]")


def flatten_and_pad(history: list[dict], max_length: int) -> tuple[list[int], list[int]]:
    """
    Flatten a tokenized user history (list of profile/event items, each with
    its own key_value_tokens) into ONE sequence of (key, value) token pairs.
    Prepends a [USR] token. Truncates if too long, pads with [PAD] if too
    short. Returns (key_ids, value_ids) as two parallel lists of equal length.

    Note: this version drops the time-encoding info (elapsed_time, hour/day
    sin-cos) -- we'll decide how to feed those back in once the basic
    flattened sequence + model is working. Flagging this as a known gap.
    """
    key_ids = [USR_KEY_ID]
    value_ids = [PAD_VALUE_ID]  # [USR] has no "value" -- placeholder

    for item in history:
        for key_id, value_id in item["key_value_tokens"]:
            key_ids.append(key_id)
            value_ids.append(value_id)

    # Truncate if too long
    key_ids = key_ids[:max_length]
    value_ids = value_ids[:max_length]

    # Pad if too short
    pad_amount = max_length - len(key_ids)
    key_ids.extend([PAD_KEY_ID] * pad_amount)
    value_ids.extend([PAD_VALUE_ID] * pad_amount)

    return key_ids, value_ids

"""
Step 9: save/load numerical boundaries to/from disk, so training and
evaluation always use the EXACT same boundaries -- computed once, reused
everywhere, rather than silently recomputed (and potentially slightly
different) each time a script runs.
"""
import json


def compute_all_boundaries(events: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    """Compute percentile boundaries for every numerical field we use."""
    return {
        "amount": compute_percentile_boundaries(events[events["type"] == "card_payment"]["amount"].tolist()),
        "fee": compute_percentile_boundaries(events["fee"].dropna().tolist()),
        "price": compute_percentile_boundaries(events[events["type"] == "trading"]["price"].tolist()),
        "balance": compute_percentile_boundaries(profiles["balance"].tolist()),
        "tenure_months": compute_percentile_boundaries(profiles["tenure_months"].tolist()),
    }


def save_boundaries(boundaries: dict, path: str = "tokenizer/boundaries.json") -> None:
    with open(path, "w") as f:
        json.dump(boundaries, f, indent=2)
    print(f"Saved boundaries to {path}")


def load_boundaries(path: str = "tokenizer/boundaries.json") -> dict:
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    print(f"Key vocabulary size: {len(ALL_KEYS)} keys")
    for key in ["type", "amount", "channel", "symbol"]:
        print(f"  '{key}' -> token id {tokenize_key(key)}")

    print(f"\nCategorical vocabularies: {len(CATEGORICAL_VOCABS)} fields")
    print(f"  type='card_payment' -> {tokenize_categorical('type', 'card_payment')}")
    print(f"  direction='sell' -> {tokenize_categorical('direction', 'sell')}")
    print(f"  region='sg' -> {tokenize_categorical('region', 'sg')}")
    print(f"  plan='metal' -> {tokenize_categorical('plan', 'metal')}")

    print("\n" + "=" * 60)
    print("Computing real boundaries from our generated dataset")
    print("=" * 60)
    import pandas as pd
    events = pd.read_parquet("data_gen/output/events.parquet")

    card_payment_amounts = events[events["type"] == "card_payment"]["amount"].tolist()
    print(f"card_payment amounts: {len(card_payment_amounts)} values, "
          f"min={min(card_payment_amounts):.2f}, max={max(card_payment_amounts):.2f}")

    real_boundaries = compute_percentile_boundaries(card_payment_amounts, num_buckets=10)
    print(f"Boundaries: {[round(b, 2) for b in real_boundaries]}")

    print("\nBucketing a few real card_payment amounts:")
    for test_value in card_payment_amounts[:5]:
        bucket = bucket_value(test_value, real_boundaries)
        print(f"  amount={test_value:.2f} -> bucket {bucket}")

    print("\n" + "=" * 60)
    print("Time encoding examples")
    print("=" * 60)
    for seconds in [1, 60, 3600, 86400, 86400 * 30, 86400 * 365]:
        encoded = encode_elapsed_time(seconds)
        print(f"  {seconds:>10} seconds ({seconds/86400:.1f} days) -> encoded: {encoded:.2f}")

    print("\n" + "=" * 60)
    print("Cyclical time encoding examples (hour of day, period=24)")
    print("=" * 60)
    for hour in [0, 6, 12, 18, 23]:
        x, y = encode_cyclical(hour, period=24)
        print(f"  hour={hour:>2} -> (x={x:.2f}, y={y:.2f})")

    print("\n" + "=" * 60)
    print("Computing fee boundaries")
    print("=" * 60)
    fee_amounts = events["fee"].dropna().tolist()
    print(f"fee values: {len(fee_amounts)} values, min={min(fee_amounts):.2f}, max={max(fee_amounts):.2f}")
    fee_boundaries = compute_percentile_boundaries(fee_amounts, num_buckets=10)
    print(f"fee boundaries: {[round(b, 2) for b in fee_boundaries]}")

    print("\n" + "=" * 60)
    print("Tokenizing two consecutive real events for one user")
    print("=" * 60)
    boundaries = {"amount": real_boundaries, "fee": fee_boundaries}

    user0_events = events[events["user_id"] == 0].sort_values("created")
    first_two = user0_events.head(2).to_dict("records")

    # First event: no previous event, so elapsed time is 0
    event_a = first_two[0]
    result_a = tokenize_event(event_a, boundaries, seconds_since_last_event=0)
    print(f"Event A (type={event_a['type']}, created={event_a['created']}):")
    print(f"  {result_a}")

    # Second event: elapsed time = gap between event A and event B
    event_b = first_two[1]
    gap_seconds = (event_b["created"] - event_a["created"]).total_seconds()
    result_b = tokenize_event(event_b, boundaries, seconds_since_last_event=gap_seconds)
    print(f"\nEvent B (type={event_b['type']}, created={event_b['created']}, gap={gap_seconds:.0f}s):")
    print(f"  {result_b}")

    print("\n" + "=" * 60)
    print("Tokenizing profile state (needs balance + tenure_months boundaries)")
    print("=" * 60)
    profiles = pd.read_parquet("data_gen/output/profiles.parquet")

    balance_boundaries = compute_percentile_boundaries(profiles["balance"].tolist(), num_buckets=10)
    tenure_boundaries = compute_percentile_boundaries(profiles["tenure_months"].tolist(), num_buckets=10)
    print(f"balance boundaries: {[round(b, 2) for b in balance_boundaries]}")
    print(f"tenure_months boundaries: {[round(b, 2) for b in tenure_boundaries]}")

    profile_boundaries = {
        "amount": real_boundaries,
        "fee": fee_boundaries,
        "balance": balance_boundaries,
        "tenure_months": tenure_boundaries,
    }

    example_profile = profiles.iloc[0].to_dict()
    profile_tokens = tokenize_profile(example_profile, profile_boundaries)
    print(f"\nProfile: {example_profile}")
    print(f"\nTokenized (key_id, value_id) pairs:")
    for key_id, value_id in profile_tokens:
        key_name = ID_TO_KEY[key_id]
        print(f"  {key_name} (key_id={key_id}) -> value_id={value_id}")

    print("\n" + "=" * 60)
    print("Tokenizing one user's FULL history (profile + all events)")
    print("=" * 60)
    user0_profile = profiles[profiles["user_id"] == 0].iloc[0].to_dict()
    user0_events = events[events["user_id"] == 0].sort_values("created").to_dict("records")

    price_values = events[events["type"] == "trading"]["price"].tolist()
    print(f"price values: {len(price_values)} values, min={min(price_values):.2f}, max={max(price_values):.2f}")
    price_boundaries = compute_percentile_boundaries(price_values, num_buckets=10)
    print(f"price boundaries: {[round(b, 2) for b in price_boundaries]}")

    full_boundaries = {
        "amount": real_boundaries,
        "fee": fee_boundaries,
        "balance": balance_boundaries,
        "tenure_months": tenure_boundaries,
        "price": price_boundaries,
    }

    history = tokenize_user_history(user0_profile, user0_events, full_boundaries)
    print(f"User 0 history: {len(history)} items (1 profile + {len(user0_events)} events)")
    print(f"\nFirst 3 items:")
    for item in history[:3]:
        print(f"  {item}")

    print("\n" + "=" * 60)
    print("Sequence lengths across many users")
    print("=" * 60)
    lengths = []
    for user_id in profiles["user_id"].head(20):
        u_profile = profiles[profiles["user_id"] == user_id].iloc[0].to_dict()
        u_events = events[events["user_id"] == user_id].sort_values("created").to_dict("records")
        u_history = tokenize_user_history(u_profile, u_events, full_boundaries)
        lengths.append(len(u_history))

    print(f"Sequence lengths for first 20 users: {lengths}")
    print(f"Min: {min(lengths)}, Max: {max(lengths)}")

    print("\n" + "=" * 60)
    print("Flatten and pad one user's history")
    print("=" * 60)
    user0_history = tokenize_user_history(user0_profile, user0_events, full_boundaries)
    key_ids, value_ids = flatten_and_pad(user0_history, max_length=250)

    print(f"Original history items: {len(user0_history)}")
    print(f"Flattened+padded length: {len(key_ids)}")
    print(f"First 10 key_ids: {key_ids[:10]}")
    print(f"First 10 value_ids: {value_ids[:10]}")
    print(f"Last 5 key_ids (should be [PAD] id={PAD_KEY_ID}): {key_ids[-5:]}")

    print("\n" + "=" * 60)
    print("Measuring flattened lengths (before padding/truncation) across users")
    print("=" * 60)
    flat_lengths = []
    for user_id in profiles["user_id"].head(50):
        u_profile = profiles[profiles["user_id"] == user_id].iloc[0].to_dict()
        u_events = events[events["user_id"] == user_id].sort_values("created").to_dict("records")
        u_history = tokenize_user_history(u_profile, u_events, full_boundaries)

        raw_length = 1  # the [USR] token
        for item in u_history:
            raw_length += len(item["key_value_tokens"])
        flat_lengths.append(raw_length)

    print(f"Flattened lengths (first 50 users): min={min(flat_lengths)}, "
          f"max={max(flat_lengths)}, avg={sum(flat_lengths)/len(flat_lengths):.1f}")
    
    print("\n" + "=" * 60)
    print("Computing and saving boundaries ONCE, for reuse everywhere")
    print("=" * 60)
    all_boundaries = compute_all_boundaries(events, profiles)
    save_boundaries(all_boundaries)

    # Confirm round-trip works
    reloaded = load_boundaries()
    print(f"Reloaded boundaries for fields: {list(reloaded.keys())}")