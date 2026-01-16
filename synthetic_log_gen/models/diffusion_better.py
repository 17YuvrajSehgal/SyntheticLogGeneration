import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import FeatureEmbedder, FeatureUnembedder

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] integer timesteps
        returns: [B, dim]
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(0, half, device=device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=device)], dim=1)
        return emb

class TransformerDenoiser(nn.Module):
    """
    Predict noise epsilon given x_t (latent) and timestep t.
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

class LogDiffusionModelBetter(nn.Module):
    """
    IMPROVED Diffusion Model with Repetition-Aware Training.
    
    Key improvements:
    1. Repetition loss to match real data statistics (279 repeats/trace)
    2. Transition frequency matching loss
    3. Pattern diversity regularization
    """
    def __init__(
        self, 
        vocab_sizes: dict, # {channel: size}
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        max_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.1,
        # NEW: Repetition-aware parameters
        target_repeats: int = 279,  # From real data statistics
        repetition_weight: float = 0.05,
        transition_weight: float = 0.03
    ):
        super().__init__()
        self.d_model = d_model
        self.max_timesteps = max_timesteps
        self.target_repeats = target_repeats
        self.repetition_weight = repetition_weight
        self.transition_weight = transition_weight
        
        # Components
        self.embedder = FeatureEmbedder(d_model, vocab_sizes, dropout)
        self.denoiser = TransformerDenoiser(d_model, nhead, num_layers, dropout)
        self.head = FeatureUnembedder(d_model, vocab_sizes)
        
        # DDPM Schedule
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, max_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Gather alpha_bars[t]
        ab = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise, noise
    
    def compute_repetition_loss(self, event_logits, target_events):
        """
        NEW: Encourage repetitive patterns like real traces.
        
        Measures how many immediate repeats (event[i] == event[i+1]) occur
        and penalizes deviation from target_repeats.
        """
        # Get predicted events (argmax of logits)
        pred_events = torch.argmax(event_logits, dim=-1)  # [B, L]
        
        # Count immediate repeats in predictions
        pred_repeats = (pred_events[:, :-1] == pred_events[:, 1:]).float().sum(dim=1)  # [B]
        
        # Count immediate repeats in targets
        target_repeats_count = (target_events[:, :-1] == target_events[:, 1:]).float().sum(dim=1)  # [B]
        
        # L1 loss to match target repetition count
        # Use target_repeats_count instead of fixed target to be adaptive
        rep_loss = F.l1_loss(pred_repeats, target_repeats_count)
        
        return rep_loss, pred_repeats.mean(), target_repeats_count.mean()
    
    def compute_transition_frequency_loss(self, event_logits, target_events):
        """
        NEW: Match transition frequency distribution.
        
        Encourages the model to generate transitions with similar frequencies
        to the training data (e.g., 7->7 should be very common).
        """
        batch_size, seq_len, vocab_size = event_logits.shape
        
        # Get predicted events
        pred_events = torch.argmax(event_logits, dim=-1)  # [B, L]
        
        # Compute transition pairs for predictions
        pred_from = pred_events[:, :-1]  # [B, L-1]
        pred_to = pred_events[:, 1:]     # [B, L-1]
        
        # Compute transition pairs for targets
        target_from = target_events[:, :-1]  # [B, L-1]
        target_to = target_events[:, 1:]     # [B, L-1]
        
        # Create transition indices (from * vocab_size + to)
        pred_transitions = pred_from * vocab_size + pred_to
        target_transitions = target_from * vocab_size + target_to
        
        # Compute histograms (normalized)
        pred_hist = torch.histc(pred_transitions.float(), bins=vocab_size*vocab_size, 
                                min=0, max=vocab_size*vocab_size-1)
        target_hist = torch.histc(target_transitions.float(), bins=vocab_size*vocab_size,
                                  min=0, max=vocab_size*vocab_size-1)
        
        # Normalize to probabilities
        pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
        target_hist = target_hist / (target_hist.sum() + 1e-8)
        
        # KL divergence loss (encourage matching distribution)
        # KL(target || pred) - we want pred to match target
        kl_loss = F.kl_div(pred_hist.log(), target_hist, reduction='batchmean')
        
        return kl_loss
    
    def forward(self, inputs: dict):
        """
        IMPROVED Training Step with Repetition Awareness.
        """
        device = next(iter(inputs.values())).device
        batch_size = next(iter(inputs.values())).shape[0]
        
        # 1. Embed
        x0 = self.embedder(inputs) # [B, L, D]
        
        # 2. Diffusion
        t = torch.randint(0, self.max_timesteps, (batch_size,), device=device).long()
        x_t, noise = self.q_sample(x0, t)
        
        # 3. Denoise
        pred_noise = self.denoiser(x_t, t)
        
        # 4. Latent Loss (MSE)
        latent_loss = F.mse_loss(pred_noise, noise)
        
        # 5. Predict x0 from eps
        ab = self.alpha_bars[t].view(-1, 1, 1)
        x0_hat = (x_t - torch.sqrt(1.0 - ab) * pred_noise) / torch.sqrt(ab)
        
        # 6. Project back to logits
        logits = self.head(x0_hat, input_keys=inputs.keys())
        
        # 7. Reconstruction Loss
        recon_loss = 0.0
        recon_loss_per_channel = {}
        
        for key, target in inputs.items():
            if key == "dt":
                # MSE for continuous
                pred = logits[key]
                channel_loss = F.mse_loss(pred, target.float())
            else:
                # Cross Entropy for discrete
                pred = logits[key].view(-1, logits[key].shape[-1])
                tgt = target.view(-1)
                channel_loss = F.cross_entropy(pred, tgt)
            
            recon_loss += channel_loss
            recon_loss_per_channel[key] = channel_loss.detach()
        
        # 8. NEW: Repetition Loss
        repetition_loss, pred_rep_count, target_rep_count = self.compute_repetition_loss(
            logits['event'], inputs['event']
        )
        
        # 9. NEW: Transition Frequency Loss
        transition_loss = self.compute_transition_frequency_loss(
            logits['event'], inputs['event']
        )
        
        # 10. Total Loss with NEW components
        total_loss = (
            latent_loss + 
            0.1 * recon_loss + 
            self.repetition_weight * repetition_loss +
            self.transition_weight * transition_loss
        )
        
        return total_loss, {
            "latent_loss": latent_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "repetition_loss": repetition_loss.detach(),
            "transition_loss": transition_loss.detach(),
            "pred_repeats": pred_rep_count.detach(),
            "target_repeats": target_rep_count.detach(),
            "recon_loss_per_channel": recon_loss_per_channel
        }
    
    @torch.no_grad()
    def sample(self, batch_size, seq_len, device):
        """
        Generate samples from pure noise.
        """
        # Start with noise
        x = torch.randn(batch_size, seq_len, self.d_model, device=device)
        
        for t in reversed(range(self.max_timesteps)):
            if t % 100 == 0:
                print(f"  [Diffusion] Denoising step {t}/{self.max_timesteps}...", end="\r")
            t_tensor = torch.full((batch_size,), t, device=device).long()
            
            # Predict noise
            eps = self.denoiser(x, t_tensor)
            
            # Update x
            beta = self.betas[t]
            alpha = self.alphas[t]
            ab = self.alpha_bars[t]
            
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - ab)) * eps)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean
                
        # Decode
        logits = self.head(x)
        outputs = {}
        for k, v in logits.items():
            if k == "dt":
                outputs[k] = v
            else:
                outputs[k] = torch.argmax(v, dim=-1)
                
        return outputs
    
    @torch.no_grad()
    def sample_ddim(self, batch_size, seq_len, device, ddim_steps=50, eta=0.0):
        """
        Fast sampling using DDIM (Denoising Diffusion Implicit Models).
        """
        # Create sampling schedule
        step_size = self.max_timesteps // ddim_steps
        timesteps = list(range(0, self.max_timesteps, step_size))
        timesteps = timesteps[:ddim_steps]
        timesteps.reverse()
        
        # Start with noise
        x = torch.randn(batch_size, seq_len, self.d_model, device=device)
        
        for i, t in enumerate(timesteps):
            if i % 10 == 0:
                print(f"  [DDIM] Step {i}/{ddim_steps} (t={t})...", end="\r")
            
            t_tensor = torch.full((batch_size,), t, device=device).long()
            
            # Predict noise
            eps = self.denoiser(x, t_tensor)
            
            # Get alpha values
            ab_t = self.alpha_bars[t]
            
            # Predict x0
            x0_pred = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
            
            if i < len(timesteps) - 1:
                # Get next timestep
                t_next = timesteps[i + 1]
                ab_next = self.alpha_bars[t_next]
                
                # DDIM update
                sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_next)
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - ab_next - sigma**2) * eps
                
                # Random noise
                noise = torch.randn_like(x) if eta > 0 else 0
                
                # Update
                x = torch.sqrt(ab_next) * x0_pred + dir_xt + sigma * noise
            else:
                # Final step
                x = x0_pred
        
        print()  # New line after progress
        
        # Decode
        logits = self.head(x)
        outputs = {}
        for k, v in logits.items():
            if k == "dt":
                outputs[k] = v
            else:
                outputs[k] = torch.argmax(v, dim=-1)
                
        return outputs
