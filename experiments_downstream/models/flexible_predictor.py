"""
Flexible Next-Event Prediction Model for Ablation Studies.

This model adapts to any subset of channels.
"""

import torch
import torch.nn as nn


class FlexibleNextEventPredictor(nn.Module):
    """
    Transformer-based predictor that works with any subset of channels.
    """
    
    def __init__(
        self,
        vocab_sizes: dict,
        channels: list,  # Which channels to use
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.channels = channels
        self.max_seq_len = max_seq_len
        
        # Create embeddings only for requested channels
        self.embeddings = nn.ModuleDict()
        emb_dims = []
        
        for ch in channels:
            if ch == 'event':
                self.embeddings[ch] = nn.Embedding(vocab_sizes['event'], d_model // 3)
                emb_dims.append(d_model // 3)
            elif ch == 'dt':
                self.embeddings[ch] = nn.Linear(1, d_model // 6)
                emb_dims.append(d_model // 6)
            elif ch in ['cpu', 'tid', 'comm', 'ret', 'fd']:
                self.embeddings[ch] = nn.Embedding(vocab_sizes[ch], d_model // 12)
                emb_dims.append(d_model // 12)
        
        # Fusion layer
        fusion_dim = sum(emb_dims)
        self.fusion = nn.Linear(fusion_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_sizes['event'])
        )
        
    def forward(self, inputs: dict):
        """
        Args:
            inputs: dict with available channels, each [B, L]
        
        Returns:
            logits: [B, num_events]
        """
        embeddings = []
        
        for ch in self.channels:
            if ch == 'dt':
                # Continuous feature
                emb = self.embeddings[ch](inputs[ch].unsqueeze(-1).float())
            else:
                # Discrete feature
                emb = self.embeddings[ch](inputs[ch])
            embeddings.append(emb)
        
        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=-1)  # [B, L, fusion_dim]
        
        # Fuse to d_model
        x = self.fusion(x)  # [B, L, d_model]
        
        # Add positional encoding
        B, L, D = x.shape
        x = x + self.pos_encoding[:, :L, :]
        
        # Transformer
        x = self.transformer(x)  # [B, L, d_model]
        
        # Use last token
        x = x[:, -1, :]  # [B, d_model]
        
        # Predict next event
        logits = self.output_head(x)  # [B, num_events]
        
        return logits
