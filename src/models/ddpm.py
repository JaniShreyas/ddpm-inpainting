import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.config import DataConfig
from src.noise_schedules import get_noise_schedule
from src.models.losses import EpsilonLoss, VLoss, XLoss, PredictionOrLossType, get_loss_function

from tqdm import tqdm


class DiffusionModel(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.config = config
        self.backbone = backbone

        # Noise Schedule
        betas = get_noise_schedule(config.model.schedule)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Just moves alphas_cumprod forward by 1, removes right most, and sets first value to 1
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # The variance values needed during sampling
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.loss_type = PredictionOrLossType(config.model.loss_type)
        self.prediction_type = PredictionOrLossType(config.model.prediction_type)

        self.loss_fn = get_loss_function(self.loss_type, alphas_cumprod, self.prediction_type)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('posterior_variance', posterior_variance)

        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod).sqrt()
        self.register_buffer('sigmas', sigmas)


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
        # t = torch.randint(0, len(self.betas), (B,), device=x.device).long()

        # From the EDM paper, sampling t using a log-normal distribution is better.
        p_mean = self.config.model.time_step_log_normal.mean
        p_std = self.config.model.time_step_log_normal.std

        rnd_normal = torch.randn((B,), device=x.device)
        sigma_target = (rnd_normal * p_std + p_mean).exp()

        # Map the continuous target sigmas to the closest discrete timestep t
        # self.sigmas is shape (T,), sigma_target is shape (B,)
        # We calculate the distance matrix (B, T) to find the closest t for each batch item
        distances = torch.abs(self.sigmas.unsqueeze(0) - sigma_target.unsqueeze(1))
        t = torch.argmin(distances, dim=1).long()

        # Add noise to images
        x_noisy, noise = self.q_sample(x_start=x, t=t)

        # Predict output
        pred = self.backbone(x, t)

        if self.loss_type == PredictionOrLossType.EPSILON:
            target = noise
        elif self.loss_type == PredictionOrLossType.V:
            sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(B, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = (1.0 - self.alphas_cumprod[t]).sqrt().view(B, 1, 1, 1)
            target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x
        elif self.loss_type == PredictionOrLossType.X:
            target = x
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        # Calculate loss
        loss = self.loss_fn(pred, target, x_noisy, t)
        return {"loss": loss}

    @torch.no_grad()
    def sample_ancestral(self, num_images, image_size, get_stats, device="cuda", **kwargs):
        """
        Inference method for generating new images
        """
        # Start with pure random noise
        x = torch.randn(
            num_images, self.backbone.in_channels, image_size, image_size, device=device
        )

        # The reverse diffusion loop. Calculate and remove noise
        for t in tqdm(
            reversed(range(0, len(self.betas))), desc="Sampling", total=len(self.betas)
        ):
            t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)

            # Predict the noise from the UNet
            model_output = self.backbone(x, t_tensor)

            # The denoising formula to get a cleaner image. From the paper
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]

            if self.prediction_type == PredictionOrLossType.EPSILON:
                predicted_noise = model_output
                
            elif self.prediction_type == PredictionOrLossType.X:
                # Model predicted clean image x_0. Solve for epsilon:
                # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
                predicted_noise = (x - alpha_cumprod_t.sqrt() * model_output) / (1.0 - alpha_cumprod_t).sqrt()
                
            elif self.prediction_type == PredictionOrLossType.V:
                # Model predicted velocity (v). Solve for epsilon mathematically:
                predicted_noise = alpha_cumprod_t.sqrt() * model_output + (1.0 - alpha_cumprod_t).sqrt() * x  
            else:
                raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

            coeff = (1 - alpha_t) / (1 - alpha_cumprod_t).sqrt()
            mean = (1 / alpha_t.sqrt()) * (x - coeff * predicted_noise)

            if t > 0:
                variance = self.posterior_variance[t].sqrt()
                z = torch.randn_like(x)
                x = mean + variance * z
            else:
                x = mean

        # Denormalize images for viewing and saving
        mean, std = get_stats(DataConfig(**self.config.dataset))
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1,-1,1,1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1,-1,1,1)
        x = x * std + mean
        x = x.clamp(0,1)
        return x
    

    # DDIM sampler
    @torch.no_grad()
    def sample(self, num_images, image_size, get_stats, device="cuda", **kwargs):
        """
        Fast, deterministic inference using DDIM.
        Mathematically stable for skipping steps in standard diffusion schedules.
        """

        num_steps = self.config.sampling.solver_num_steps

        # Create a linear spacing of your discrete timesteps
        total_steps = len(self.betas)
        step_indices = torch.linspace(total_steps - 1, 0, num_steps, dtype=torch.long, device=device)

        # Start with pure random noise
        x = torch.randn(
            num_images, self.backbone.in_channels, image_size, image_size, device=device
        )

        # The DDIM loop
        for i in tqdm(range(len(step_indices)), desc=f"DDIM Sampling ({num_steps} steps)"):
            t = step_indices[i]
            # Get the exact timestep we are jumping to. If at the end, drop to -1.
            t_prev = step_indices[i + 1] if i < len(step_indices) - 1 else -1

            t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)

            # Predict the output from the UNet
            model_output = self.backbone(x, t_tensor)

            # Look up alpha_bar for current step and the step we are jumping to
            alpha_cumprod_t = self.alphas_cumprod[t]
            # If t_prev is -1, we are at the fully clean image, so alpha_bar is 1.0
            alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # Calculate both predicted noise AND predicted clean image (x0)
            if self.prediction_type == PredictionOrLossType.EPSILON:
                predicted_noise = model_output
                pred_x0 = (x - (1.0 - alpha_cumprod_t).sqrt() * predicted_noise) / alpha_cumprod_t.sqrt()
                
            elif self.prediction_type == PredictionOrLossType.X:
                pred_x0 = model_output
                predicted_noise = (x - alpha_cumprod_t.sqrt() * pred_x0) / (1.0 - alpha_cumprod_t).sqrt()
                
            elif self.prediction_type == PredictionOrLossType.V:
                predicted_noise = alpha_cumprod_t.sqrt() * model_output + (1.0 - alpha_cumprod_t).sqrt() * x
                pred_x0 = alpha_cumprod_t.sqrt() * x - (1.0 - alpha_cumprod_t).sqrt() * model_output
                
            else:
                raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

            # =================================================================
            # The Core DDIM Update Step
            # Point straight to x0, then strictly add back the required noise 
            # for the specific timestep we are jumping to.
            # =================================================================
            direction_pointing_to_xt = (1.0 - alpha_cumprod_t_prev).sqrt() * predicted_noise
            x = alpha_cumprod_t_prev.sqrt() * pred_x0 + direction_pointing_to_xt

        # Denormalize images for viewing and saving
        mean, std = get_stats(DataConfig(**self.config.dataset))
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        x = x * std + mean
        x = x.clamp(0, 1)
        
        return x
