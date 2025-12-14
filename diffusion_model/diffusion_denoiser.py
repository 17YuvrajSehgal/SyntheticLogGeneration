import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] integer timesteps
        returns: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(0, half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device)], dim=1)
        return emb

class TransformerDenoiser(nn.Module):
    """
    Predict noise epsilon given x_t and timestep t.
    x_t: [B, L, d_model]
    t:   [B]
    out: [B, L, d_model]
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L, D = x_t.shape
        te = self.time_mlp(self.time_emb(t)).unsqueeze(1)  # [B, 1, D]
        h = x_t + te
        h = self.encoder(h)
        return self.out(h)
