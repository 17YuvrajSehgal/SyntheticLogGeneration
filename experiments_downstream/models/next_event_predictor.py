"""
Next-Event Prediction Model for Downstream Task Evaluation.

This model predicts the next event given a sequence of kernel trace events.
Used to evaluate the utility of synthetic data for downstream tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NextEventPredictor(nn.Module):
    """
    Transformer-based next-event prediction model.
    
    Input: Sequence of events with metadata (event, dt, cpu, tid, comm, ret)
    Output: Probability distribution over next event (384 classes)
    """
    
    def __init__(
        self,
        vocab_sizes: dict,  # {channel: size}
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Feature embeddings
        self.event_emb = nn.Embedding(vocab_sizes['event'], d_model // 6)
        self.cpu_emb = nn.Embedding(vocab_sizes['cpu'], d_model // 12)
        self.tid_emb = nn.Embedding(vocab_sizes['tid'], d_model // 12)
        self.comm_emb = nn.Embedding(vocab_sizes['comm'], d_model // 12)
        self.ret_emb = nn.Embedding(vocab_sizes['ret'], d_model // 12)
        
        # Time delta projection (continuous)
        self.dt_proj = nn.Linear(1, d_model // 12)
        
        # Fusion layer
        fusion_dim = (d_model // 6) + 5 * (d_model // 12)
        self.fusion = nn.Linear(fusion_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head (predict next event)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_sizes['event'])
        )
        
    def forward(self, inputs: dict):
        """
        Args:
            inputs: dict with keys ['event', 'dt', 'cpu', 'tid', 'comm', 'ret']
                    Each tensor has shape [B, L]
        
        Returns:
            logits: [B, num_events] - logits for next event prediction
        """
        # Embed each feature
        event_emb = self.event_emb(inputs['event'])  # [B, L, d/6]
        cpu_emb = self.cpu_emb(inputs['cpu'])        # [B, L, d/12]
        tid_emb = self.tid_emb(inputs['tid'])        # [B, L, d/12]
        comm_emb = self.comm_emb(inputs['comm'])     # [B, L, d/12]
        ret_emb = self.ret_emb(inputs['ret'])        # [B, L, d/12]
        dt_emb = self.dt_proj(inputs['dt'].unsqueeze(-1).float())  # [B, L, d/12]
        
        # Concatenate all features
        x = torch.cat([event_emb, dt_emb, cpu_emb, tid_emb, comm_emb, ret_emb], dim=-1)
        
        # Fuse to d_model
        x = self.fusion(x)  # [B, L, d_model]
        
        # Add positional encoding
        B, L, D = x.shape
        x = x + self.pos_encoding[:, :L, :]
        
        # Transformer encoding
        x = self.transformer(x)  # [B, L, d_model]
        
        # Use last token for prediction
        x = x[:, -1, :]  # [B, d_model]
        
        # Predict next event
        logits = self.output_head(x)  # [B, num_events]
        
        return logits


class NextEventPredictorEventOnly(nn.Module):
    """
    Simplified version that only uses event IDs (for ablation study).
    """
    
    def __init__(
        self,
        num_events: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Event embedding
        self.event_emb = nn.Embedding(num_events, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer encoder
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
            nn.Linear(d_model, num_events)
        )
        
    def forward(self, event_ids: torch.Tensor):
        """
        Args:
            event_ids: [B, L] - sequence of event IDs
        
        Returns:
            logits: [B, num_events]
        """
        # Embed events
        x = self.event_emb(event_ids)  # [B, L, d_model]
        
        # Add positional encoding
        B, L, D = x.shape
        x = x + self.pos_encoding[:, :L, :]
        
        # Transformer encoding
        x = self.transformer(x)  # [B, L, d_model]
        
        # Use last token
        x = x[:, -1, :]  # [B, d_model]
        
        # Predict next event
        logits = self.output_head(x)  # [B, num_events]
        
        return logits
