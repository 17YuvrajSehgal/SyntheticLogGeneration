import torch
import torch.nn as nn

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0, t, noise=None):
        """
        x0: [B, L, D]
        t:  [B] in [0, T-1]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return a * x0 + b * noise, noise

def train_one_step(model, diffusion, x0, optimizer):
    """
    Predict noise epsilon at random t and compute MSE.
    """
    model.train()
    B = x0.shape[0]
    t = torch.randint(0, diffusion.T, (B,), device=x0.device, dtype=torch.long)
    x_t, noise = diffusion.q_sample(x0, t)
    pred = model(x_t, t)
    loss = nn.functional.mse_loss(pred, noise)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()
