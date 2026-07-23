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

if __name__ == "__main__":
    embed_dim = 32
    num_keys = 22
    num_values = 10
    max_length = 250

    embedding_layer = TokenEmbedding(num_keys, num_values, embed_dim)
    position_layer = PositionalEmbedding(max_length, embed_dim)
    encoder = MiniPragmaEncoder(embed_dim, num_heads=4, num_layers=2)

    fake_key_ids = torch.tensor([[1, 5, 6, 0, 0]])
    fake_value_ids = torch.tensor([[-1, 2, 0, -1, -1]])
    fake_padding_mask = torch.tensor([[False, False, False, True, True]])  # last 2 are [PAD]

    x = embedding_layer(fake_key_ids, fake_value_ids)
    x = position_layer(x)
    output = encoder(x, padding_mask=fake_padding_mask)

    print(f"Encoder output shape: {output.shape}")
    print(f"(Should still be [1, 5, {embed_dim}] -- Transformer preserves shape, just mixes info between tokens)")

    user_embedding = pool_usr_token(output)
    print(f"\nUser embedding shape: {user_embedding.shape}")
    print(f"(Should be [1, {embed_dim}] -- ONE vector per user, not per token)")