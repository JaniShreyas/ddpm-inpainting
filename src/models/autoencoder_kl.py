import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.blocks import ResidualBlock, AttentionBlock

# Should contain an Encoder and Decoder class with config as input


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        double_z: bool,
        base_channels: int,
        channel_multipliers: list[int],
        attention_resolutions: list[int],
        num_res_blocks: int,
        dropout: int,
        image_size: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        self.num_res_blocks = num_res_blocks

        self.image_size = image_size

        if double_z:
            self.latent_channels = 2 * latent_channels
        else:
            self.latent_channels = latent_channels


        # Initial convolution to base_channels
        self.initial_conv = nn.Conv2d(self.in_channels, self.base_channels, 3, padding=1)

        channels = [self.base_channels] + [self.base_channels*channel_multipliers[i] for i in range(len(channel_multipliers))]
        current_res = self.image_size

        self.down = nn.Sequential()

        # m blocks of (Residual + Downsample)
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i+1]
            
            downs = []
            for _ in range(self.num_res_blocks):
                downs.append(ResidualBlock(in_ch, out_ch, dropout=dropout))
                in_ch=out_ch
            if current_res in attention_resolutions:
                downs.append(AttentionBlock(out_ch))

            downs.append(nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1))
            
            current_res //= 2  # Since we always half the resolution

            self.down.add_module(f"{i}", nn.Sequential(*downs))
        
        # Residual, Attention, Residual
        self.bottleneck = nn.Sequential(
            ResidualBlock(in_channels=channels[-1], out_channels=channels[-1], dropout=dropout),
            AttentionBlock(channels[-1]),
            ResidualBlock(in_channels=channels[-1], out_channels=channels[-1], dropout=dropout),
        )

        # GroupNorm, SiLU, Conv2D to latent_channels (will be given double z or not by the VAE class)
        # 32 groups is a standard heuristic choice
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=channels[-1], eps=1e-6, affine=True)
        self.silu = nn.SiLU()
        self.latent = nn.Conv2d(channels[-1], self.latent_channels, kernel_size=3, padding=1)


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
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)
