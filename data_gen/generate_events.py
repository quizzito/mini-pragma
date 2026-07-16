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
    Combine one user's profile state and event history into a single record
    — this is the full "user timeline" concept from Figure 2 of the paper.
    """
    profile = generate_profile(user_id=user_id)
    events = generate_event_history(num_events=num_events)
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

if __name__ == "__main__":
    records = generate_all_users(num_users=100)

    event_counts = [len(r["events"]) for r in records]
    print(f"Generated {len(records)} users.")
    print(f"Event counts — min: {min(event_counts)}, max: {max(event_counts)}, avg: {sum(event_counts)/len(event_counts):.1f}")

    # Peek at one user to confirm shape is still correct
    print("\nExample user (id=0):")
    print(records[0]["profile"])
    print(f"...with {len(records[0]['events'])} events")