"""
M5 training loop for mini-PRAGMA.

Step 1: single-batch overfit test. Before training on real, varied data,
we check the model CAN learn at all by repeatedly training on the exact
same batch and confirming loss drops toward ~0. If this fails, something
is wrong with the model/loss wiring -- no point moving to real training
until this passes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from tokenizer.event_tokenizer import load_boundaries, get_max_value_id, ALL_KEYS
from model.dataset import UserHistoryDataset
from model.mini_pragma import MiniPragma, MLMHead, mask_values


def train_step(model, mlm_head, batch, mask_token_id, num_values, optimizer):
    """Run one training step: mask, forward, loss, backward, optimizer update."""
    optimizer.zero_grad()

    masked_values, mask_positions = mask_values(batch["value_ids"], mask_token_id, mask_prob=0.15)
    padding_mask = batch["key_ids"] == 0

    encoder_output, _ = model(batch["key_ids"], masked_values, padding_mask)
    predictions = mlm_head(encoder_output)

    flat_predictions = predictions.view(-1, num_values)
    flat_true_values = batch["value_ids"].view(-1)
    flat_mask_positions = mask_positions.view(-1)

    masked_predictions = flat_predictions[flat_mask_positions]
    masked_true_values = flat_true_values[flat_mask_positions]

    loss = F.cross_entropy(masked_predictions, masked_true_values)
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    profiles = pd.read_parquet("data_gen/output/profiles.parquet")
    events = pd.read_parquet("data_gen/output/events.parquet")
    boundaries = load_boundaries()

    dataset = UserHistoryDataset(profiles, events, boundaries, max_length=250)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))  # grab ONE batch, we'll reuse it repeatedly

    num_keys = len(ALL_KEYS)
    num_values = get_max_value_id() + 1
    mask_token_id = num_values

    model = MiniPragma(num_keys=num_keys, num_values=num_values + 1, embed_dim=32, max_length=250)
    mlm_head = MLMHead(embed_dim=32, num_values=num_values)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mlm_head.parameters()), lr=1e-3
    )

    # Mask ONCE, outside the loop -- so we're overfitting a FIXED task,
    # not chasing a different random mask every step
    masked_values, mask_positions = mask_values(batch["value_ids"], mask_token_id, mask_prob=0.15)
    padding_mask = batch["key_ids"] == 0

    print("Overfitting a single batch (fixed mask) -- loss should drop toward ~0:")
    for step in range(200):
        optimizer.zero_grad()

        encoder_output, _ = model(batch["key_ids"], masked_values, padding_mask)
        predictions = mlm_head(encoder_output)

        flat_predictions = predictions.view(-1, num_values)
        flat_true_values = batch["value_ids"].view(-1)
        flat_mask_positions = mask_positions.view(-1)

        masked_predictions = flat_predictions[flat_mask_positions]
        masked_true_values = flat_true_values[flat_mask_positions]

        loss = F.cross_entropy(masked_predictions, masked_true_values)
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 199:
            print(f"  step {step:>3}: loss = {loss.item():.4f}")