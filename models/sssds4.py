"""
SSSDS4 (Structured State Space Diffusion S4) model implementation.
Based on the ICSE 2025 paper: "Execution Trace Reconstruction Using Diffusion-Based Generative Models"

This implementation adapts the SSSDS4 model for unconditional generation of kernel traces.
"""

import torch
import torch.nn as nn
import numpy as np
from .s4 import S4Block


class DiffusionScheduler:
    """Linear noise schedule for diffusion process."""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta = np.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = np.cumprod(self.alpha)
        self.alpha_cumprod_prev = np.concatenate([[1.0], self.alpha_cumprod[:-1]])
        
    def add_noise(self, x, t):
        """Add noise to data at timestep t."""
        if isinstance(t, torch.Tensor):
            t_np = t.cpu().numpy()
            sqrt_alpha_cumprod = torch.tensor(np.sqrt(self.alpha_cumprod[t_np]), device=x.device, dtype=x.dtype)
            sqrt_one_minus_alpha_cumprod = torch.tensor(np.sqrt(1.0 - self.alpha_cumprod[t_np]), device=x.device, dtype=x.dtype)
            # Expand dimensions if needed
            if len(sqrt_alpha_cumprod.shape) < len(x.shape):
                for _ in range(len(x.shape) - len(sqrt_alpha_cumprod.shape)):
                    sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
                    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        else:
            sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod[t])
            sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod[t])
            if isinstance(x, torch.Tensor):
                sqrt_alpha_cumprod = torch.tensor(sqrt_alpha_cumprod, device=x.device, dtype=x.dtype)
                sqrt_one_minus_alpha_cumprod = torch.tensor(sqrt_one_minus_alpha_cumprod, device=x.device, dtype=x.dtype)
                if len(sqrt_alpha_cumprod.shape) < len(x.shape):
                    for _ in range(len(x.shape) - len(sqrt_alpha_cumprod.shape)):
                        sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
                        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        noise = torch.randn_like(x)
        noisy_x = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_x, noise
    
    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class SSSDS4(nn.Module):
    """
    SSSDS4 model for generating kernel trace sequences.
    Combines S4 layers with diffusion process for sequence generation.
    """
    
    def __init__(self, 
                 input_dim=1,
                 d_model=128,
                 num_layers=4,
                 num_timesteps=1000,
                 dropout=0.1):
        """
        Args:
            input_dim: Input dimension (1 for univariate sequences)
            d_model: Model dimension
            num_layers: Number of S4 blocks
            num_timesteps: Number of diffusion timesteps
            dropout: Dropout rate
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.d_model = d_model
        
        # Input projection (input_dim for data + d_model for timestep embedding)
        self.input_proj = nn.Linear(input_dim + d_model, d_model)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # S4 blocks
        self.s4_blocks = nn.ModuleList([
            S4Block(d_model, dropout) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, input_dim)
        )
        
        # Initialize timestep embedding
        self._init_timestep_embedding()
    
    def _init_timestep_embedding(self):
        """Initialize sinusoidal timestep embedding."""
        # This will be used to create positional encodings for timesteps
        pass
    
    def get_timestep_embedding(self, timesteps):
        """Create sinusoidal timestep embeddings."""
        half_dim = self.d_model // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb
    
    def forward(self, x, timesteps):
        """
        Forward pass through the model.
        
        Args:
            x: Noisy input of shape (batch, seq_len, input_dim)
            timesteps: Timestep indices of shape (batch,)
        
        Returns:
            Predicted noise of shape (batch, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Get timestep embeddings
        t_emb = self.get_timestep_embedding(timesteps)
        t_emb = self.time_embed(t_emb)  # (batch, d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, d_model)
        
        # Concatenate input with timestep embedding
        x_proj = self.input_proj(torch.cat([x, t_emb], dim=-1))
        
        # Pass through S4 blocks
        h = x_proj
        for s4_block in self.s4_blocks:
            h = s4_block(h)
        
        # Output projection
        output = self.output_proj(h)
        
        return output


class TraceGenerator:
    """Wrapper class for generating traces using SSSDS4 model."""
    
    def __init__(self, model, scheduler, device='cuda'):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(self, seq_len=200, batch_size=1, num_samples=1):
        """
        Generate synthetic kernel trace sequences.
        
        Args:
            seq_len: Length of sequences to generate
            batch_size: Batch size for generation
            num_samples: Number of samples to generate
        
        Returns:
            Generated sequences of shape (num_samples, seq_len, 1)
        """
        all_samples = []
        
        for _ in range(num_samples):
            # Start with pure noise
            x = torch.randn(batch_size, seq_len, 1, device=self.device)
            
            # Reverse diffusion process
            for t in range(self.scheduler.num_timesteps - 1, -1, -1):
                timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.model(x, timesteps)
                
                # Denoise step using DDPM sampling
                alpha_t = torch.tensor(self.scheduler.alpha[t], device=self.device, dtype=torch.float32)
                alpha_cumprod_t = torch.tensor(self.scheduler.alpha_cumprod[t], device=self.device, dtype=torch.float32)
                
                if t > 0:
                    alpha_cumprod_t_prev = torch.tensor(self.scheduler.alpha_cumprod_prev[t], device=self.device, dtype=torch.float32)
                    beta_t = torch.tensor(self.scheduler.beta[t], device=self.device, dtype=torch.float32)
                else:
                    alpha_cumprod_t_prev = torch.tensor(1.0, device=self.device, dtype=torch.float32)
                    beta_t = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                
                # Compute predicted x_0
                pred_x0 = (x - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                
                # Compute direction pointing to x_t
                pred_dir_xt = torch.sqrt(1.0 - alpha_cumprod_t_prev) * predicted_noise
                
                # Update x
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt
                
                # Add noise for next step (except last step)
                if t > 0:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise
            
            all_samples.append(x.cpu().numpy())
        
        return np.concatenate(all_samples, axis=0)

