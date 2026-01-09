import torch
import torch.nn as nn

class TraceEmbedding(nn.Module):
    """
    Converts categorical channels (event_id, dt_bucket, cpu_id) into a continuous
    sequence embedding suitable for diffusion denoisers.

    Input:  x_ids [B, L, 3]  (int64)
            channels = [event, dt, cpu]
    Output: x_emb [B, L, d_model] (float32)
    """

    def __init__(
        self,
        num_events: int,
        num_dt_buckets: int = 256,
        num_cpus: int = 256,  # safe upper bound; you can set exact later
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Split the model dimension across channels (simple + effective)
        d_event = d_model // 2
        d_dt    = d_model // 4
        d_cpu   = d_model - d_event - d_dt

        self.event_emb = nn.Embedding(num_events, d_event)
        self.dt_emb    = nn.Embedding(num_dt_buckets, d_dt)
        self.cpu_emb   = nn.Embedding(num_cpus, d_cpu)

        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        # x_ids: [B, L, 3]
        event = x_ids[..., 0]
        dt    = x_ids[..., 1]
        cpu   = x_ids[..., 2]

        x = torch.cat([
            self.event_emb(event),
            self.dt_emb(dt),
            self.cpu_emb(cpu),
        ], dim=-1)  # [B, L, d_model]

        x = self.proj(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x
