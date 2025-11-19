"""
S4 (Structured State Space) layer implementation.
Based on the S4 model from the ICSE 2025 paper.
"""

import torch
import torch.nn as nn
import math


class S4Layer(nn.Module):
    """
    Structured State Space (S4) layer for modeling long-term dependencies.
    This is a simplified implementation based on the S4 architecture.
    """
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # State space parameters
        # A: (d_model, d_model) - state transition matrix
        # B: (d_model, d_model) - input-to-state matrix
        # C: (d_model, d_model) - state-to-output matrix  
        # D: scalar - direct input-to-output connection
        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.D = nn.Parameter(torch.randn(1) * 0.01)
        
        # Normalization and activation
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply state space transformation
        # Simplified version: using linear transformation
        # Full S4 would use more complex state space operations with HiPPO initialization
        output = torch.zeros_like(x)
        
        # State space computation (simplified)
        # Initialize state
        state = torch.zeros(batch_size, self.d_model, device=x.device, dtype=x.dtype)
        
        for t in range(seq_len):
            # State update: x' = Ax + Bu
            # state: (batch, d_model)
            # A: (d_model, d_model)
            # input_t: (batch, d_model)
            # B: (d_model, d_model)
            input_t = x[:, t, :]  # (batch, d_model)
            state = torch.matmul(state, self.A.T) + torch.matmul(input_t, self.B.T)
            
            # Output: y = Cx + Du
            # state: (batch, d_model)
            # C: (d_model, d_model)
            # input_t: (batch, d_model)
            # D: scalar
            output[:, t, :] = torch.matmul(state, self.C.T) + self.D * input_t
        
        output = self.norm(output)
        output = self.dropout(output)
        
        return output


class S4Block(nn.Module):
    """S4 block with residual connection."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.s4 = S4Layer(d_model, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # S4 layer with residual
        x = x + self.s4(x)
        # Feed-forward with residual
        x = x + self.ff(self.norm(x))
        return x

