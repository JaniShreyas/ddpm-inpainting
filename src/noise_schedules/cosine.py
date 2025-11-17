import torch
import math

def cosine_noise_schedule(T: int, s: float, max_beta: float = 0.999, name: str = 'cosine'):
    t = torch.linspace(0, T, T+1) / T
    # the cosine schedule formula
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    # normalize so that alpha_bar(0) == 1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # betas from discrete differences of alpha_bar
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=0.0, max=max_beta)
    return betas
