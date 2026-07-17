"""
Data quality validation pass for the synthetic mini-pragma dataset.
Checks: distribution sanity, label correlation with intended drivers,
and basic data integrity (nulls, duplicates).
"""
import pandas as pd

profiles = pd.read_parquet("data_gen/output/profiles.parquet")
events = pd.read_parquet("data_gen/output/events.parquet")

print("=" * 60)
print("1. BASIC INTEGRITY CHECKS")
print("=" * 60)
print(f"Profiles: {len(profiles)} rows, {profiles['user_id'].nunique()} unique user_ids")
print(f"Duplicate user_ids: {profiles['user_id'].duplicated().sum()}")
print(f"Null values in profiles:\n{profiles.isnull().sum()}")
print(f"\nEvents: {len(events)} rows")
print(f"Users with zero events: {profiles['user_id'].nunique() - events['user_id'].nunique()}")
print(f"Null values in events (per column, NaN is expected for type-specific fields):")
print(events.isnull().sum())

print("\n" + "=" * 60)
print("2. DISTRIBUTION SANITY CHECKS")
print("=" * 60)
print("\nBalance distribution:")
print(profiles["balance"].describe())

print("\nTenure distribution (months):")
print(profiles["tenure_months"].describe())

events_per_user = events.groupby("user_id").size()
print("\nEvents per user distribution:")
print(events_per_user.describe())

print("\nEvent type breakdown (should be roughly even across 5 types, ~20% each):")
print(events["type"].value_counts(normalize=True))

print("\n" + "=" * 60)
print("3. LABEL CORRELATION CHECKS (does each label depend on what we intended?)")
print("=" * 60)

# credit_default was designed to depend on spend-to-balance ratio and tenure
card_spend = events[events["type"] == "card_payment"].groupby("user_id")["amount"].sum()
profiles["total_card_spend"] = profiles["user_id"].map(card_spend).fillna(0)
profiles["spend_to_balance"] = profiles["total_card_spend"] / (profiles["balance"] + 1)

print("\ncredit_default vs spend_to_balance ratio (should be higher for defaulters):")
print(profiles.groupby("credit_default")["spend_to_balance"].mean())

print("\ncredit_default vs tenure_months (should be lower for defaulters):")
print(profiles.groupby("credit_default")["tenure_months"].mean())

# fraud was designed to depend on event velocity and large trades
print("\nfraud vs total events per user (should be somewhat higher for fraud=1):")
events_count = events.groupby("user_id").size()
profiles["event_count"] = profiles["user_id"].map(events_count).fillna(0)
print(profiles.groupby("fraud")["event_count"].mean())

# engagement was designed to depend on app activity and prior comms interaction
app_events = events[events["type"] == "app_event"].groupby("user_id").size()
profiles["app_event_count"] = profiles["user_id"].map(app_events).fillna(0)
print("\nengagement vs app_event_count (should be higher for engagement=1):")
print(profiles.groupby("engagement")["app_event_count"].mean())

print("\n" + "=" * 60)
print("4. RED-FLAG CHECK: label correlation with user_id (should be ~no relationship)")
print("=" * 60)
print(profiles[["user_id", "credit_default", "fraud", "engagement"]].corr()["user_id"])