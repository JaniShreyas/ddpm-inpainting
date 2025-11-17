import torch

def linear_noise_schedule(beta_start: float, beta_end: float, T: int, name: str = 'linear'):
    return torch.linspace(beta_start, beta_end, T)