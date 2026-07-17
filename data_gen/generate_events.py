"""
Step 1 of the synthetic data generator: a single user's PROFILE STATE.

Profile state = static/slowly-changing attributes, captured at one point in time
(e.g. "as of today, this user is on the metal plan, based in the UK, balance $6,012").
This mirrors §2.1.2 of the PRAGMA paper.

This is deliberately the simplest possible piece: one function, one user, no
events yet, no randomness loop over thousands of users. We'll build those next.
"""
import random
import pandas as pd
from datetime import datetime, timedelta

def generate_profile(user_id: int) -> dict:
    """Generate one synthetic user's profile state (a snapshot, not a history)."""
    region = random.choice(["uk", "us", "eu", "au", "sg"])
    plan = random.choice(["standard", "plus", "premium", "metal", "ultra"])
    tenure_months = random.randint(1, 60)  # how long they've been a customer
    balance = round(random.uniform(50, 20000), 2)
    age_bracket = random.choice(["18-24", "25-34", "35-44", "45-54", "55+"])

    return {
        "user_id": user_id,
        "region": region,
        "plan": plan,
        "tenure_months": tenure_months,
        "balance": balance,
        "age_bracket": age_bracket,
    }

def generate_users(num_users: int) -> list[dict]:
    """Generate profile states for many users. Just loops generate_profile."""
    return [generate_profile(user_id=i) for i in range(num_users)]

def generate_card_payment(timestamp: datetime) -> dict:
    """Generate one synthetic card_payment event, mirroring Figure 2/3 in the paper."""
    merchant_categories = ["groceries", "restaurant", "transport", "shopping", "subscription"]
    category = random.choice(merchant_categories)

    return {
        "created": timestamp,
        "type": "card_payment",
        "direction": "out",
        "amount": round(random.uniform(2, 200), 2),
        "currency": "gbp",
        "fee": 0.0,
        "description": category,
    }

def generate_topup(timestamp: datetime) -> dict:
    """Generate one synthetic topup event (money added to the account)."""
    return {
        "created": timestamp,
        "type": "topup",
        "direction": "in",
        "amount": round(random.uniform(20, 500), 2),
        "currency": "gbp",
        "fee": 0.0,
    }

def generate_app_event(timestamp: datetime) -> dict:
    """Generate one synthetic app_event (in-app navigation, no money involved)."""
    screens = ["home", "p2p_amount", "confirm_p2p_dialog", "card_settings", "statements", "support_chat"]
    return {
        "created": timestamp,
        "type": "app_event",
        "view": random.choice(screens),
    }

def generate_communication(timestamp: datetime) -> dict:
    """Generate one synthetic communication event (email/push notification)."""
    channels = ["email", "push", "sms"]
    products = ["credit_card", "savings", "stocks_shares_isa", "current_account"]
    interactions = ["sent", "opened", "interacted"]
    return {
        "created": timestamp,
        "type": "communication",
        "channel": random.choice(channels),
        "product": random.choice(products),
        "interact": random.choice(interactions),
    }

def generate_trading(timestamp: datetime) -> dict:
    """Generate one synthetic trading event (buy/sell order)."""
    symbols = ["swda", "vwrl", "aapl", "tsla", "btc"]
    return {
        "created": timestamp,
        "type": "trading",
        "direction": random.choice(["buy", "sell"]),
        "symbol": random.choice(symbols),
        "amount": random.randint(1, 100),
        "price": round(random.uniform(10, 500), 4),
        "currency": "gbx",
        "order_type": random.choice(["market", "limit"]),
    }

def generate_event_history(num_events: int, days_back: int = 90) -> list[dict]:
    """
    Generate a sequence of events for ONE user, spread randomly over the last
    `days_back` days, sorted chronologically (oldest first). Randomly mixes
    event types, since real users generate heterogeneous event streams.
    """
    now = datetime.now()
    events = []
    for _ in range(num_events):
        random_offset = timedelta(
            days=random.uniform(0, days_back),
            seconds=random.randint(0, 86400),
        )
        event_time = now - random_offset

        event_type = random.choice(
            ["card_payment", "topup", "app_event", "communication", "trading"]
        )
        if event_type == "card_payment":
            events.append(generate_card_payment(timestamp=event_time))
        elif event_type == "topup":
            events.append(generate_topup(timestamp=event_time))
        elif event_type == "app_event":
            events.append(generate_app_event(timestamp=event_time))
        elif event_type == "communication":
            events.append(generate_communication(timestamp=event_time))
        else:
            events.append(generate_trading(timestamp=event_time))

    events.sort(key=lambda e: e["created"])
    return events

def generate_user_record(user_id: int, num_events: int = 20) -> dict:
    """
    Combine one user's profile state and event history into a single record.
    Attaches all 3 synthetic downstream-task labels to the profile:
    credit_default, fraud, and engagement.
    """
    profile = generate_profile(user_id=user_id)
    events = generate_event_history(num_events=num_events)
    profile["credit_default"] = generate_credit_default_label(profile, events)
    profile["fraud"] = generate_fraud_label(events)
    profile["engagement"] = generate_engagement_label(events)
    return {
        "profile": profile,
        "events": events,
    }

def generate_all_users(num_users: int, min_events: int = 5, max_events: int = 50) -> list[dict]:
    """
    Generate full records (profile + event history) for many users.
    Event count varies per user, since real users have very different
    activity levels (matches the paper's long-tailed event-count observation).
    """
    records = []
    for user_id in range(num_users):
        num_events = random.randint(min_events, max_events)
        records.append(generate_user_record(user_id=user_id, num_events=num_events))
    return records

def save_records(records: list[dict], output_dir: str = "data_gen/output") -> None:
    """
    Save generated user records as two flat tables:
    - profiles.parquet: one row per user
    - events.parquet: one row per event, with a user_id column linking back
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    profiles_df = pd.DataFrame([r["profile"] for r in records])

    all_events = []
    for r in records:
        for event in r["events"]:
            event_with_user = {"user_id": r["profile"]["user_id"], **event}
            all_events.append(event_with_user)
    events_df = pd.DataFrame(all_events)

    profiles_df.to_parquet(f"{output_dir}/profiles.parquet", index=False)
    events_df.to_parquet(f"{output_dir}/events.parquet", index=False)

    print(f"Saved {len(profiles_df)} profiles to {output_dir}/profiles.parquet")
    print(f"Saved {len(events_df)} events to {output_dir}/events.parquet")

def compute_user_spend(events: list[dict]) -> float:
    """Sum up all card_payment amounts for one user's event list (a simple
    behavioral feature we'll use to simulate a credit default label)."""
    total = 0.0
    for e in events:
        if e["type"] == "card_payment":
            total += e["amount"]
    return round(total, 2)

def compute_default_probability(profile: dict, total_spend: float) -> float:
    """
    Simulate a probability of credit default based on behavior + profile,
    plus noise. This is a MADE-UP rule for synthetic data generation —
    not a real credit model. Higher spend relative to balance -> higher risk.
    Longer tenure -> slightly lower risk (established customer).
    """
    balance = profile["balance"]
    tenure = profile["tenure_months"]

    spend_to_balance_ratio = total_spend / (balance + 1)  # +1 avoids divide-by-zero

    base_risk = 0.05  # everyone has some baseline default risk
    risk_from_spend = min(spend_to_balance_ratio * 0.3, 0.5)  # capped so it can't dominate
    risk_reduction_from_tenure = min(tenure / 60 * 0.05, 0.05)  # long-tenure users slightly safer

    probability = base_risk + risk_from_spend - risk_reduction_from_tenure
    noise = random.uniform(-0.05, 0.05)  # the "plus noise" part — keeps it imperfectly predictable

    probability = probability + noise
    return max(0.0, min(1.0, probability))  # clip to a valid [0, 1] probability

def generate_credit_default_label(profile: dict, events: list[dict]) -> int:
    """Convert a simulated default probability into a binary label (0 or 1)."""
    spend = compute_user_spend(events)
    prob = compute_default_probability(profile, spend)
    return 1 if random.random() < prob else 0

def compute_event_velocity(events: list[dict]) -> float:
    """
    Rough proxy for 'unusual activity burst' — events per day, based on the
    span between first and last event. Used as a fraud signal: a sudden
    flurry of activity is a classic (simplified) fraud heuristic.
    """
    if len(events) < 2:
        return 0.0
    span_days = (events[-1]["created"] - events[0]["created"]).total_seconds() / 86400
    if span_days < 0.1:
        span_days = 0.1  # avoid divide-by-zero for very tight clusters
    return len(events) / span_days


def compute_fraud_probability(events: list[dict]) -> float:
    """
    Simulate fraud probability from event velocity + large trading amounts.
    MADE-UP rule for synthetic data — fraud is deliberately kept rare (~2-4%),
    matching real-world class imbalance for this task.
    """
    velocity = compute_event_velocity(events)
    large_trades = sum(1 for e in events if e["type"] == "trading" and e["amount"] > 70)

    base_risk = 0.01
    risk_from_velocity = min(velocity * 0.01, 0.15)
    risk_from_large_trades = min(large_trades * 0.03, 0.15)

    probability = base_risk + risk_from_velocity + risk_from_large_trades
    noise = random.uniform(-0.02, 0.02)
    probability += noise
    return max(0.0, min(1.0, probability))


def generate_fraud_label(events: list[dict]) -> int:
    """Convert simulated fraud probability into a binary label."""
    prob = compute_fraud_probability(events)
    return 1 if random.random() < prob else 0


def compute_engagement_probability(events: list[dict]) -> float:
    """
    Simulate probability that a user engages with (opens) a re-engagement
    communication. MADE-UP rule: more active app users and users who've
    previously interacted with comms are more likely to engage again.
    """
    app_event_count = sum(1 for e in events if e["type"] == "app_event")
    prior_interactions = sum(
        1 for e in events if e["type"] == "communication" and e["interact"] == "interacted"
    )

    base_rate = 0.15
    boost_from_app_activity = min(app_event_count * 0.01, 0.25)
    boost_from_prior_comms = min(prior_interactions * 0.08, 0.25)

    probability = base_rate + boost_from_app_activity + boost_from_prior_comms
    noise = random.uniform(-0.05, 0.05)
    probability += noise
    return max(0.0, min(1.0, probability))


def generate_engagement_label(events: list[dict]) -> int:
    """Convert simulated engagement probability into a binary label."""
    prob = compute_engagement_probability(events)
    return 1 if random.random() < prob else 0

if __name__ == "__main__":
    records = generate_all_users(num_users=2000)
    save_records(records)

    profiles_df = pd.read_parquet("data_gen/output/profiles.parquet")
    print(profiles_df[["user_id", "credit_default", "fraud", "engagement"]].head(10))
    print(f"\nCredit default rate: {profiles_df['credit_default'].mean():.1%}")
    print(f"Fraud rate: {profiles_df['fraud'].mean():.1%}")
    print(f"Engagement rate: {profiles_df['engagement'].mean():.1%}")