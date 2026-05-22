import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.config import DataConfig
from tqdm import tqdm

class JiTFlowModel(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.config = config
        self.backbone = backbone

        # Log-Normal sampling parameters (from config, defaults match JiT)
        # JiT uses mean = -0.8, std = 0.8
        self.p_mean = getattr(config.model, 'time_step_log_normal_mean', -0.8)
        self.p_std = getattr(config.model, 'time_step_log_normal_std', 0.8)
        self.t_eps = 1e-5 # For clamping division by zero

    def sample_t(self, batch_size, device):
        """
        JiT's Log-Normal timestep sampler mapped through a sigmoid.
        Biases training to spend more time in the middle of the flow.
        """
        z = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        return torch.sigmoid(z)

    def forward(self, x, **kwargs):
        cond = kwargs.get('cond')
        B, C, H, W = x.shape
        device = x.device

        t = self.sample_t(B, device=device).view(B, 1, 1, 1)
        e = torch.randn_like(x)

        z_t = t * x + (1.0 - t) * e
        v_true = (x - z_t) / (1.0 - t).clamp_min(self.t_eps)

        model_input = torch.cat([z_t, cond], dim=1) if cond is not None else z_t
        
        x_pred = self.backbone(model_input, t.flatten())
        
        if cond is not None and x_pred.shape[1] != C:
            x_pred = x_pred[:, :C, :, :]

        v_pred = (x_pred - z_t) / (1.0 - t).clamp_min(self.t_eps)
        loss = F.mse_loss(v_pred, v_true)
        
        return {"loss": loss}

    @torch.no_grad()
    def _get_velocity(self, z_t, t_tensor, cond):
        model_input = torch.cat([z_t, cond], dim=1) if cond is not None else z_t
        x_pred = self.backbone(model_input, t_tensor)
        
        if cond is not None and x_pred.shape[1] != z_t.shape[1]:
            x_pred = x_pred[:, :z_t.shape[1], :, :]
            
        t_expand = t_tensor.view(-1, 1, 1, 1)
        v_pred = (x_pred - z_t) / (1.0 - t_expand).clamp_min(self.t_eps)
        
        return v_pred

    @torch.no_grad()
    def sample(self, num_images, image_size, get_stats, device="cuda", **kwargs):
        cond = kwargs.get('cond')
        num_steps = self.config.sampling.solver_num_steps
        method = getattr(self.config.sampling, 'method', 'heun').lower()
        
        out_channels = self.backbone.in_channels if cond is None else self.backbone.in_channels - cond.shape[1]

        z_t = torch.randn(num_images, out_channels, image_size, image_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for i in tqdm(range(num_steps), desc=f"{method.capitalize()} Flow ({num_steps} steps)"):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr
            
            t_tensor = torch.full((num_images,), t_curr, device=device)
            t_next_tensor = torch.full((num_images,), t_next, device=device)

            if method == "heun" and i < num_steps - 1:
                v_pred_t = self._get_velocity(z_t, t_tensor, cond)
                z_next_euler = z_t + dt * v_pred_t
                v_pred_t_next = self._get_velocity(z_next_euler, t_next_tensor, cond)
                v_pred = 0.5 * (v_pred_t + v_pred_t_next)
                z_t = z_t + dt * v_pred
            else:
                v_pred = self._get_velocity(z_t, t_tensor, cond)
                z_t = z_t + dt * v_pred

        x = z_t
        mean, std = get_stats(DataConfig(**self.config.dataset))
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        
        x = x * std + mean
        x = x.clamp(0, 1)
        
        return x