"""
mini-PRAGMA model, v1 (flattened architecture).

Step 1: token embeddings. Each token in our flattened sequence has a
key_id and a value_id (e.g. key="amount", value="bucket 6"). Following
the paper's equation 1 (§2.3.1), we embed each separately then SUM them --
this lets the model learn "amount" means something different combined
with bucket 6 vs bucket 2, without needing a separate embedding for every
possible (key, value) combination.
"""
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, num_keys: int, num_values: int, embed_dim: int):
        super().__init__()
        self.key_embedding = nn.Embedding(num_keys, embed_dim)
        # +1 for our PAD_VALUE_ID=-1, which we'll shift to 0 -- see forward()
        self.value_embedding = nn.Embedding(num_values + 1, embed_dim)

    def forward(self, key_ids: torch.Tensor, value_ids: torch.Tensor) -> torch.Tensor:
        # value_ids can be -1 (our PAD_VALUE_ID) or -- for [USR] tokens --
        # also -1 as a placeholder. nn.Embedding can't handle -1 directly
        # (embeddings are indexed 0, 1, 2, ...), so shift -1 -> 0 and treat
        # index 0 of the value embedding table as "no value" going forward.
        safe_value_ids = value_ids.clone()
        safe_value_ids[safe_value_ids == -1] = 0

        key_vecs = self.key_embedding(key_ids)
        value_vecs = self.value_embedding(safe_value_ids)

        return key_vecs + value_vecs  # equation 1: sum, not concatenation


if __name__ == "__main__":
    # Quick shape test with fake small numbers, before plugging in real data
    embed_dim = 32
    num_keys = 22    # matches our tokenizer's ALL_KEYS length
    num_values = 10  # placeholder -- we'll figure out the real max later

    embedding_layer = TokenEmbedding(num_keys, num_values, embed_dim)

    fake_key_ids = torch.tensor([[1, 5, 6, 0, 0]])     # 1 user, 5 tokens (0=[PAD])
    fake_value_ids = torch.tensor([[-1, 2, 0, -1, -1]])  # matching values

    output = embedding_layer(fake_key_ids, fake_value_ids)
    print(f"Input shape: {fake_key_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [1, 5, {embed_dim}] -- (1 user, 5 tokens, {embed_dim}-dim vectors each)")