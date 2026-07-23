"""
PyTorch Dataset for mini-PRAGMA (flattened v1).

A PyTorch Dataset's job is simple: given an index (e.g. user #47), return
that user's data in tensor form. PyTorch's DataLoader then handles grabbing
many indices at once and stacking them into a batch -- we don't have to
write that batching logic ourselves.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset
import pandas as pd

from tokenizer.event_tokenizer import (
    tokenize_user_history,
    flatten_and_pad,
    compute_percentile_boundaries,
)


class UserHistoryDataset(Dataset):
    def __init__(self, profiles: pd.DataFrame, events: pd.DataFrame, boundaries: dict, max_length: int = 250):
        self.profiles = profiles
        self.events = events
        self.boundaries = boundaries
        self.max_length = max_length
        self.user_ids = profiles["user_id"].tolist()

    def __len__(self):
        # PyTorch calls this to know how many items are in the dataset
        return len(self.user_ids)

    def __getitem__(self, idx):
        # PyTorch calls this with an index, expects one user's data back
        user_id = self.user_ids[idx]
        profile = self.profiles[self.profiles["user_id"] == user_id].iloc[0].to_dict()
        user_events = (
            self.events[self.events["user_id"] == user_id]
            .sort_values("created")
            .to_dict("records")
        )

        history = tokenize_user_history(profile, user_events, self.boundaries)
        key_ids, value_ids = flatten_and_pad(history, self.max_length)

        return {
            "key_ids": torch.tensor(key_ids, dtype=torch.long),
            "value_ids": torch.tensor(value_ids, dtype=torch.long),
        }


if __name__ == "__main__":
    # Quick manual test: build boundaries from a small sample, create the
    # dataset, and check that indexing into it works and produces the
    # right tensor shapes.
    profiles = pd.read_parquet("data_gen/output/profiles.parquet")
    events = pd.read_parquet("data_gen/output/events.parquet")

    amount_vals = events[events["type"] == "card_payment"]["amount"].tolist()
    fee_vals = events["fee"].dropna().tolist()
    price_vals = events[events["type"] == "trading"]["price"].tolist()
    balance_vals = profiles["balance"].tolist()
    tenure_vals = profiles["tenure_months"].tolist()

    boundaries = {
        "amount": compute_percentile_boundaries(amount_vals),
        "fee": compute_percentile_boundaries(fee_vals),
        "price": compute_percentile_boundaries(price_vals),
        "balance": compute_percentile_boundaries(balance_vals),
        "tenure_months": compute_percentile_boundaries(tenure_vals),
    }

    dataset = UserHistoryDataset(profiles, events, boundaries, max_length=250)
    print(f"Dataset size: {len(dataset)} users")

    item = dataset[0]
    print(f"\nItem 0:")
    print(f"  key_ids shape: {item['key_ids'].shape}")
    print(f"  value_ids shape: {item['value_ids'].shape}")
    print(f"  key_ids dtype: {item['key_ids'].dtype}")

    print("\n" + "=" * 60)
    print("Testing DataLoader batching")
    print("=" * 60)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    print(f"Batch key_ids shape: {batch['key_ids'].shape}")
    print(f"Batch value_ids shape: {batch['value_ids'].shape}")