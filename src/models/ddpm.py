import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


class DiffusionModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.unet = UNet(**model_config)

        # Noise Schedule
        T = 200
        beta_start = 1e-4
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, T)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(x_start.shape[0], 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1. - self.alphas_cumprod[t]).sqrt().view(x_start.shape[0], 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def forward(self, x):
        B,C,H,W = x.shape
        # Uniformly sample t
        t = torch.randint(0, len(self.betas), (B,), device=x.device).long()

        # Add noise to images
        x_noisy, noise = self.q_sample(x_start=x, t=t)

        # Predict noise
        predicted_noise = self.unet(x_noisy, t)

        # Calculate loss
        loss = F.mse_loss(noise, predicted_noise)
        return loss