# autoregressive_transformer.py
import torch
import torch.nn as nn

def causal_mask(L: int, device: torch.device):
    # mask[i, j] = -inf if j > i (prevent attending to future)
    m = torch.full((L, L), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)
    return m

class ARTransformer(nn.Module):
    """
    Autoregressive Transformer over token triplets (event, dt, cpu).
    Input:  x_ids [B, L, 3] (int64)
    Output: logits dict over positions [B, L, V] for each channel
    """
    def __init__(self, embed_module: nn.Module, d_model: int, nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embed = embed_module  # TraceEmbedding
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x_ids: torch.Tensor):
        """
        x_ids: [B, L, 3]
        returns hidden states: [B, L, D]
        """
        x = self.embed(x_ids)  # [B, L, D]
        L = x.shape[1]
        mask = causal_mask(L, x.device)  # [L, L]
        h = self.encoder(x, mask=mask)
        return h
