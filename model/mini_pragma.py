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
    
class PositionalEmbedding(nn.Module):
    """
    Learnable position embedding -- one vector per sequence position,
    added to the token embedding so the model knows token order.
    Simpler than the paper's RoPE (deferred per our M5 scope decision).
    """
    def __init__(self, max_length: int, embed_dim: int):
        super().__init__()
        self.position_embedding = nn.Embedding(max_length, embed_dim)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        seq_length = token_embeddings.shape[1]
        positions = torch.arange(seq_length, device=token_embeddings.device)
        position_vecs = self.position_embedding(positions)
        return token_embeddings + position_vecs  # broadcasts across the batch

class MiniPragmaEncoder(nn.Module):
    """
    The actual Transformer -- stacks a few TransformerEncoderLayers.
    batch_first=True keeps our tensor shape convention [batch, seq, dim]
    throughout, matching everything we've built so far.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 2, ff_dim: int = 128):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        # padding_mask: True where a position IS padding (to be ignored)
        return self.transformer(x, src_key_padding_mask=padding_mask)
    
def pool_usr_token(encoder_output: torch.Tensor) -> torch.Tensor:
    """
    Extract the [USR] token's final vector -- always position 0 -- as the
    single summary embedding for the whole user. Same role as z_h,0 in the
    paper's Figure 4.
    """
    return encoder_output[:, 0, :]  # [batch, seq, dim] -> [batch, dim]

class MiniPragma(nn.Module):
    """
    The full mini-PRAGMA model (flattened v1): wires together everything
    we've built -- token embedding, positional embedding, Transformer
    encoder, and USR-token pooling -- into one forward pass.
    """
    def __init__(
        self,
        num_keys: int,
        num_values: int,
        embed_dim: int = 32,
        max_length: int = 250,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(num_keys, num_values, embed_dim)
        self.position_embedding = PositionalEmbedding(max_length, embed_dim)
        self.encoder = MiniPragmaEncoder(embed_dim, num_heads, num_layers, ff_dim)

    def forward(self, key_ids: torch.Tensor, value_ids: torch.Tensor, padding_mask: torch.Tensor = None):
        x = self.token_embedding(key_ids, value_ids)
        x = self.position_embedding(x)
        encoder_output = self.encoder(x, padding_mask=padding_mask)
        user_embedding = pool_usr_token(encoder_output)
        return encoder_output, user_embedding

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pandas as pd
    from torch.utils.data import DataLoader
    from tokenizer.event_tokenizer import load_boundaries, get_max_value_id, ALL_KEYS
    from model.dataset import UserHistoryDataset

    profiles = pd.read_parquet("data_gen/output/profiles.parquet")
    events = pd.read_parquet("data_gen/output/events.parquet")
    boundaries = load_boundaries()

    dataset = UserHistoryDataset(profiles, events, boundaries, max_length=250)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    num_keys = len(ALL_KEYS)
    num_values = get_max_value_id() + 1  # +1 since ids are 0-indexed

    model = MiniPragma(num_keys=num_keys, num_values=num_values, embed_dim=32, max_length=250)

    padding_mask = batch["key_ids"] == 0  # True where key_id == [PAD]'s id (0)
    encoder_output, user_embedding = model(batch["key_ids"], batch["value_ids"], padding_mask)

    print(f"Batch size: {batch['key_ids'].shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"User embedding shape: {user_embedding.shape}")
    print(f"(Expect: encoder_output=[8, 250, 32], user_embedding=[8, 32])")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,}")