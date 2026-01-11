import torch
import torch.nn as nn
import math

class FeatureEmbedder(nn.Module):
    """
    Embeds multiple discrete/continuous features into a common latent dimension.
    """
    def __init__(
        self,
        d_model: int,
        vocab_sizes: dict, # {channel_name: size}
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Discrete Embeddings
        self.embeddings = nn.ModuleDict()
        
        # We handle these keys if present in input
        self.discrete_keys = ["event", "cpu", "tid", "fd", "comm", "ret", "file"]
        
        for key in self.discrete_keys:
            if key in vocab_sizes:
                vocab_size = vocab_sizes[key]
                # Prefix to avoid collision with nn.Module attributes (e.g. 'cpu')
                module_key = f"emb_{key}"
                self.embeddings[module_key] = nn.Embedding(vocab_size, d_model)

        # Continuous Input (dt)
        # We project scalar dt -> d_model
        self.dt_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        inputs: dict of {key: Tensor[B, L]}
        returns: Tensor[B, L, d_model] (Sum of embeddings)
        """
        batch_size = next(iter(inputs.values())).shape[0]
        seq_len = next(iter(inputs.values())).shape[1]
        device = next(iter(inputs.values())).device
        
        # Start with zeros
        x_sum = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Add Discrete
        for key in self.discrete_keys:
            module_key = f"emb_{key}"
            if key in inputs and module_key in self.embeddings:
                emb = self.embeddings[module_key](inputs[key]) # [B, L, D]
                x_sum = x_sum + emb
                
        # Add Continuous (dt)
        if "dt" in inputs:
            dt = inputs["dt"].unsqueeze(-1) # [B, L, 1]
            dt_emb = self.dt_proj(dt)
            x_sum = x_sum + dt_emb
            
        return self.dropout(x_sum)


class FeatureUnembedder(nn.Module):
    """
    Projects latent state back to feature logits.
    """
    def __init__(self, d_model: int, vocab_sizes: dict):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.vocab_sizes = vocab_sizes
        
        # Discrete Heads
        for key, size in vocab_sizes.items():
            if key != "dt":
               module_key = f"head_{key}"
               self.heads[module_key] = nn.Linear(d_model, size)
        
        # Continuous Head (dt)
        if "dt" in vocab_sizes: 
             pass
        self.dt_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, input_keys: list = None) -> dict:
        """
        x: [B, L, D]
        returns: dict of logits
        """
        outputs = {}
        
        # If no keys specified, do all
        if input_keys:
             keys_to_run = input_keys
        else:
             # Infer from heads, but heads keys have prefix
             keys_to_run = [k.replace("head_", "") for k in self.heads.keys()]
             keys_to_run.append("dt")
        
        for key in keys_to_run:
            module_key = f"head_{key}"
            if module_key in self.heads:
                outputs[key] = self.heads[module_key](x)
            elif key == "dt":
                outputs[key] = self.dt_head(x).squeeze(-1)
                
        return outputs
