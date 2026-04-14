import torch
import torch.nn as nn
from tqdm import tqdm
from src.data.config import DataConfig
from src.models.losses.edm_loss import EDMLoss 

class EDMModel(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.config = config
        
        # This backbone is now expected to be the EDMPrecond wrapper 
        # (which has the SongUNet inside it)
        self.backbone = backbone

        self.loss_fn = EDMLoss(**config.model.time_step_log_normal)

    def forward(self, x):
        """
        The EDMLoss class handles the training:
        1. Samples the continuous noise level (sigma) from a log-normal distribution.
        2. Adds that noise to 'x'.
        3. Passes it through the preconditioned backbone.
        4. Calculates the weighted MSE loss.
        """
        # The NVlabs EDMLoss expects the network and the clean images
        loss = self.loss_fn(net=self.backbone, images=x)
        
        return {"loss": loss.mean()}

    @torch.no_grad()
    def sample(self, num_images, image_size, get_stats, device="cuda", **kwargs):
        """
        Inference using the EDM Deterministic 2nd-Order Heun Solver.
        This replaces both Ancestral and DDIM sampling. It is faster and achieves 
        state-of-the-art FID in roughly 35 to 50 steps.
        """
        num_steps = self.config.sampling.solver_num_steps 

        # EDM Paper default hyperparameters for the continuous noise schedule
        sigma_min = 0.002
        sigma_max = 80.0
        rho = 7.0

        # Create the polynomial noise schedule (Equation 5 in the paper)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        
        # Append 0 at the end to represent the fully clean image
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        # Start with pure random noise scaled by the maximum noise level
        x = torch.randn(
            num_images, self.backbone.img_channels, image_size, image_size, device=device
        ) * t_steps[0]

        # The Heun Solver Loop
        for i in tqdm(range(num_steps), desc=f"EDM Heun Sampling ({num_steps} steps)"):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]

            # --- Euler Step (Evaluate the ODE at current step) ---
            # Create a tensor of the current noise level for the whole batch
            sigma_tensor = torch.full((num_images,), t_cur, device=device)
            
            # The preconditioned network predicts the clean image directly (D_theta)
            denoised = self.backbone(x, sigma_tensor)
            
            # Calculate the derivative (direction pointing to the clean image)
            d_cur = (x - denoised) / t_cur
            
            # Take an Euler step towards the next noise level
            x_next = x + (t_next - t_cur) * d_cur

            # --- Heun Correction Step (2nd-Order refinement) ---
            if t_next > 0:
                sigma_next_tensor = torch.full((num_images,), t_next, device=device)
                denoised_next = self.backbone(x_next, sigma_next_tensor)
                
                # Calculate the derivative at the proposed next step
                d_next = (x_next - denoised_next) / t_next
                
                # Average the two derivatives for a much more accurate step
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)

            x = x_next

        # Denormalize images for viewing and saving (Kept from your pipeline)
        mean, std = get_stats(DataConfig(**self.config.dataset))
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        x = x * std + mean
        x = x.clamp(0, 1)
        
        return x