import torch
import torch.nn.functional as F

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

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # posterior variance for sampling
        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]], dim=0)
        self.posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return a * x0 + b * noise, noise

    def predict_x0_from_eps(self, x_t, t, eps):
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        x0_hat = (x_t - b * eps) / (a + 1e-8)
        return x0_hat

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        One reverse step: x_t -> x_{t-1}
        model(x_t, t) returns eps prediction
        """
        B = x_t.shape[0]
        eps = model(x_t, t)

        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        abar_t = self.alpha_bars[t].view(-1, 1, 1)

        # DDPM mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - abar_t)) * eps)

        # noise (except at t=0)
        var = self.posterior_var[t].view(-1, 1, 1)
        noise = torch.randn_like(x_t)
        mask = (t != 0).float().view(-1, 1, 1)
        return mean + mask * torch.sqrt(var + 1e-8) * noise


def diffusion_step_losses(denoiser, diffusion, x0, t):
    """
    Returns:
      mse_loss, x0_hat, x_t
    """
    x_t, noise = diffusion.q_sample(x0, t)
    pred_noise = denoiser(x_t, t)
    mse = F.mse_loss(pred_noise, noise)
    x0_hat = diffusion.predict_x0_from_eps(x_t, t, pred_noise)
    return mse, x0_hat, x_t