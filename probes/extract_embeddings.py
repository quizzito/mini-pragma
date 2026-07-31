"""
M6 step 1: load the pretrained checkpoint and confirm it loads correctly.
This is a real check, not a formality -- if model hyperparameters
(embed_dim, num_keys, etc.) don't exactly match what was used during
training, loading either fails outright or silently loads garbage.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizer.event_tokenizer import ALL_KEYS, get_max_value_id
from model.mini_pragma import MiniPragma, MLMHead

CHECKPOINT_PATH = "results/checkpoints/mini_pragma_pretrained.pt"

# These MUST match exactly what was used during training in model/train.py
num_keys = len(ALL_KEYS)
num_values = get_max_value_id() + 1
embed_dim = 32
max_length = 250

model = MiniPragma(num_keys=num_keys, num_values=num_values + 1, embed_dim=embed_dim, max_length=max_length)
mlm_head = MLMHead(embed_dim=embed_dim, num_values=num_values)

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
mlm_head.load_state_dict(checkpoint["mlm_head_state_dict"])

model.eval()  # switches off dropout etc. -- we want deterministic inference, not training behavior

print("Checkpoint loaded successfully.")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

## Step 2: extract embedding for a single user (user_id=0) and sanity-check the shape and values.

import pandas as pd
from tokenizer.event_tokenizer import load_boundaries, tokenize_user_history, flatten_and_pad

boundaries = load_boundaries()
profiles = pd.read_parquet("data_gen/output/profiles.parquet")
events = pd.read_parquet("data_gen/output/events.parquet")

user0_profile = profiles[profiles["user_id"] == 0].iloc[0].to_dict()
user0_events = events[events["user_id"] == 0].sort_values("created").to_dict("records")

history = tokenize_user_history(user0_profile, user0_events, boundaries)
key_ids, value_ids = flatten_and_pad(history, max_length=max_length)

key_ids_tensor = torch.tensor([key_ids], dtype=torch.long)     # add batch dim -> [1, 250]
value_ids_tensor = torch.tensor([value_ids], dtype=torch.long)
padding_mask = key_ids_tensor == 0

with torch.no_grad():  # no gradients needed -- we're not training, just extracting
    encoder_output, user_embedding = model(key_ids_tensor, value_ids_tensor, padding_mask)

print(f"\nUser 0 embedding shape: {user_embedding.shape}")
print(f"User 0 embedding (first 8 values): {user_embedding[0, :8].tolist()}")

## Step 3  

print("\n" + "=" * 60)
print("Extracting embeddings for ALL users")
print("=" * 60)

all_embeddings = []
user_ids_in_order = profiles["user_id"].tolist()

for user_id in user_ids_in_order:
    profile = profiles[profiles["user_id"] == user_id].iloc[0].to_dict()
    user_events = events[events["user_id"] == user_id].sort_values("created").to_dict("records")

    history = tokenize_user_history(profile, user_events, boundaries)
    key_ids, value_ids = flatten_and_pad(history, max_length=max_length)

    key_ids_tensor = torch.tensor([key_ids], dtype=torch.long)
    value_ids_tensor = torch.tensor([value_ids], dtype=torch.long)
    padding_mask = key_ids_tensor == 0

    with torch.no_grad():
        _, user_embedding = model(key_ids_tensor, value_ids_tensor, padding_mask)

    all_embeddings.append(user_embedding[0].numpy())

import numpy as np
embeddings_array = np.array(all_embeddings)
print(f"Extracted embeddings for {len(all_embeddings)} users")
print(f"Embeddings array shape: {embeddings_array.shape}")

os.makedirs("results", exist_ok=True)
np.save("results/user_embeddings.npy", embeddings_array)
np.save("results/user_embeddings_user_ids.npy", np.array(user_ids_in_order))
print("Saved embeddings to results/user_embeddings.npy")