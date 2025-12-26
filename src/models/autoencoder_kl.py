import torch
import torch.nn as nn
import torch.nn.functional as F

# Should contain an Encoder and Decoder class with config as input


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        base_channels: int,
        channel_multipliers: list[int],
        attention_resolutions: list[int],
        time_emb_dim: int,
        num_res_blocks: int,
        dropout: int,
    ):
        pass

    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        base_channels: int,
        channel_multipliers: list[int],
        attention_resolutions: list[int],
        time_emb_dim: int,
        num_res_blocks: int,
        dropout: int,
    ):
        pass

    def forward(self, x):
        pass


# Combines the Encoder and Decoder
# returns loss in forward() and has a sample() method
class AutoEncoderKL(nn.Module):
    def __init__(
        self,
        config,
    ):
        self.encoder = Encoder(**config)

        self.decoder = Decoder(**config)

    def forward(self, x):

        # Encode
        mu, log_var = self.encode(x)

        # KL Loss
        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1,2,3]))

        # Reparameterization trick
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(mu)
        z = mu + std * eps

        # Decode
        x_hat = self.decode(z)

        # Reconstruction loss
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]

        total_loss = kl_loss + reconstruction_loss
        
        return total_loss
    

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)

    def decode(self, z):
        return self.decoder(z)
