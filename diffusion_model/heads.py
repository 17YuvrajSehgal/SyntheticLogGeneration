import torch.nn as nn

class TokenHeads(nn.Module):
    """
    Map denoised embeddings -> logits for event/dt/cpu.
    """
    def __init__(self, d_model, num_events, num_dt_buckets=256, num_cpus=4):
        super().__init__()
        self.event_head = nn.Linear(d_model, num_events)
        self.dt_head    = nn.Linear(d_model, num_dt_buckets)
        self.cpu_head   = nn.Linear(d_model, num_cpus)

    def forward(self, x):  # x: [B, L, D]
        return {
            "event": self.event_head(x),
            "dt":    self.dt_head(x),
            "cpu":   self.cpu_head(x),
        }
