import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet

from tqdm import tqdm


class DiffusionModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.unet = UNet(**model_config)

        # Noise Schedule
        T = 200
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Just moves alphas_cumprod forward by 1, removes right most, and sets first value to 1
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # The variance values needed during sampling
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = (
            self.alphas_cumprod[t].sqrt().view(x_start.shape[0], 1, 1, 1)
        )
        sqrt_one_minus_alphas_cumprod_t = (
            (1.0 - self.alphas_cumprod[t]).sqrt().view(x_start.shape[0], 1, 1, 1)
        )

        return (
            sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise,
            noise,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Uniformly sample t
        t = torch.randint(0, len(self.betas), (B,), device=x.device).long()

        # Add noise to images
        x_noisy, noise = self.q_sample(x_start=x, t=t)

        # Predict noise
        predicted_noise = self.unet(x_noisy, t)

        # Calculate loss
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def sample(self, num_images, image_size, device="cuda"):
        """
        Inference method for generating new images
        """
        # Start with pure random noise
        x = torch.randn(
            num_images, self.unet.in_channels, image_size, image_size, device=device
        )

        # The reverse diffusion loop. Calculate and remove noise
        for t in tqdm(
            reversed(range(0, len(self.betas))), desc="Sampling", total=len(self.betas)
        ):
            t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)

            # Predict the noise from the UNet
            predicted_noise = self.unet(x, t_tensor)

            # The denoising formula to get a cleaner image. From the paper
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]

            beta_t = self.betas[t]

            coeff = (1 - alpha_t) / (1 - alpha_cumprod_t).sqrt()
            mean = (1 / alpha_t.sqrt()) * (x - coeff * predicted_noise)

            if t > 0:
                variance = self.posterior_variance[t].sqrt()
                z = torch.randn_like(x)
                x = mean + variance * z
            else:
                x = mean

        # Denormalize images from [-1, 1] to [0, 1] for viewing and saving
        x = (x.clamp(-1, 1) + 1) / 2
        return x
