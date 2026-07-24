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

def mask_values(value_ids: torch.Tensor, mask_token_id: int, mask_prob: float = 0.15):
    """
    Randomly mask ~mask_prob of value tokens for the MLM objective.
    Returns (masked_value_ids, mask_positions) -- mask_positions is True
    wherever we masked something, so we know what to compute loss on.
    Never masks [PAD] positions (value_id == -1) -- nothing real there to predict.
    """
    is_paddable = value_ids != -1  # True where there's a real value to potentially mask
    random_mask = torch.rand(value_ids.shape, device=value_ids.device) < mask_prob
    mask_positions = random_mask & is_paddable

    masked_value_ids = value_ids.clone()
    masked_value_ids[mask_positions] = mask_token_id

    return masked_value_ids, mask_positions
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
    
###  Build a prediction head

### Now we need to actually predict what was hidden — using the model's output at each masked position, 
### run it through a small linear layer that outputs a probability distribution over all possible value ids (0-9),
### then compare to the true value.

class MLMHead(nn.Module):
    """
    Masked Language Model prediction head. Takes the encoder's per-token
    output and predicts which value_id was originally there, for every
    position (we'll only compute loss on the masked ones).
    """
    def __init__(self, embed_dim: int, num_values: int):
        super().__init__()
        self.predictor = nn.Linear(embed_dim, num_values)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.predictor(encoder_output)  # [batch, seq, num_values] -- logits per position

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

    print("\n" + "=" * 60)
    print("Testing masking")
    print("=" * 60)
    mask_token_id = num_values  # one id beyond our real values, reserved for [MASK]
    masked_values, mask_positions = mask_values(batch["value_ids"], mask_token_id, mask_prob=0.15)

    print(f"Original value_ids (first user, first 15 tokens): {batch['value_ids'][0, :15].tolist()}")
    print(f"Masked value_ids   (first user, first 15 tokens): {masked_values[0, :15].tolist()}")
    print(f"Mask positions     (first user, first 15 tokens): {mask_positions[0, :15].tolist()}")
    print(f"\nTotal masked positions in batch: {mask_positions.sum().item()} out of {mask_positions.numel()} tokens")

    print("\n" + "=" * 60)
    print("Testing MLM head")
    print("=" * 60)
    mlm_head = MLMHead(embed_dim=32, num_values=num_values)

    # Run the model on the MASKED input, then predict
    encoder_output_masked, _ = model(batch["key_ids"], masked_values, padding_mask)
    predictions = mlm_head(encoder_output_masked)

    print(f"Predictions shape: {predictions.shape}")
    print(f"(Expect: [8, 250, {num_values}] -- for every position, a score for each possible value 0-{num_values-1})")

    print("\n" + "=" * 60)
    print("Computing MLM loss")
    print("=" * 60)
    import torch.nn.functional as F

    # Flatten batch+sequence dims together, since cross_entropy expects
    # [N, num_classes] predictions vs [N] true labels
    flat_predictions = predictions.view(-1, num_values)         # [8*250, 10]
    flat_true_values = batch["value_ids"].view(-1)               # [8*250]
    flat_mask_positions = mask_positions.view(-1)                 # [8*250]

    # Only keep the masked positions -- that's what we actually score
    masked_predictions = flat_predictions[flat_mask_positions]
    masked_true_values = flat_true_values[flat_mask_positions]

    loss = F.cross_entropy(masked_predictions, masked_true_values)
    print(f"Number of masked positions being scored: {flat_mask_positions.sum().item()}")
    print(f"MLM loss: {loss.item():.4f}")