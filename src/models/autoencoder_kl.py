import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.data import denormalize
from src.data.config import DataConfig
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
        dropout: float,
        image_size: int,
        **kwargs,
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
        self.initial_conv = nn.Conv2d(
            self.in_channels, self.base_channels, 3, padding=1
        )

        channels = [self.base_channels] + [
            self.base_channels * channel_multipliers[i]
            for i in range(len(channel_multipliers))
        ]
        current_res = self.image_size

        self.down = nn.Sequential()

        # m blocks of (Residual + Downsample)
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            downs = []
            for _ in range(self.num_res_blocks):
                downs.append(ResidualBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch
            if current_res in attention_resolutions:
                downs.append(AttentionBlock(out_ch))

            downs.append(nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1))

            current_res //= 2  # Since we always half the resolution

            self.down.add_module(f"{i}", nn.Sequential(*downs))

        # Residual, Attention, Residual
        self.bottleneck = nn.Sequential(
            ResidualBlock(
                in_channels=channels[-1], out_channels=channels[-1], dropout=dropout
            ),
            AttentionBlock(channels[-1]),
            ResidualBlock(
                in_channels=channels[-1], out_channels=channels[-1], dropout=dropout
            ),
        )

        # GroupNorm, SiLU, Conv2D to latent_channels (will be given double z or not by the VAE class)
        # 32 groups is a standard heuristic choice
        self.group_norm = nn.GroupNorm(
            num_groups=32, num_channels=channels[-1], eps=1e-6, affine=True
        )
        self.silu = nn.SiLU()
        self.latent = nn.Conv2d(
            channels[-1], self.latent_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.down(x)
        x = self.bottleneck(x)
        x = self.latent(self.silu(self.group_norm(x)))
        return x


class Decoder(nn.Module):
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
        dropout: float,
        image_size: int,
        **kwargs,
    ):
        # The opposite of the Encoder just assuming the input is reparameterized (so z channels)
        # The channel multipliers go in the opposite direction obviously

        super().__init__()

        self.out_channels = out_channels
        self.base_channels = base_channels

        self.num_res_blocks = num_res_blocks

        self.image_size = image_size

        channels = [self.base_channels] + [
            self.base_channels * channel_multipliers[i]
            for i in range(len(channel_multipliers))
        ]

        self.initial_conv = nn.Conv2d(
            in_channels=latent_channels,
            out_channels=channels[-1],
            kernel_size=3,
            padding=1,
        )

        # Residual, Attention, Residual
        self.bottleneck = nn.Sequential(
            ResidualBlock(
                in_channels=channels[-1], out_channels=channels[-1], dropout=dropout
            ),
            AttentionBlock(channels[-1]),
            ResidualBlock(
                in_channels=channels[-1], out_channels=channels[-1], dropout=dropout
            ),
        )

        self.up = nn.Sequential()
        reversed_channels = list(reversed(channels))

        current_res = self.image_size // (2 ** (len(channel_multipliers)))

        # m blocks of (Residual + Upsample)
        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]

            ups = []
            for _ in range(self.num_res_blocks):
                ups.append(ResidualBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch
            if current_res in attention_resolutions:
                ups.append(AttentionBlock(out_ch))

            ups.append(nn.Upsample(scale_factor=2, mode="nearest"))
            ups.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

            current_res *= 2  # Since we always double the resolution

            self.up.add_module(f"{i}", nn.Sequential(*ups))

        self.group_norm = nn.GroupNorm(
            num_groups=32, num_channels=channels[0], eps=1e-6, affine=True
        )
        self.silu = nn.SiLU()
        self.output = nn.Conv2d(
            channels[0], self.out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bottleneck(x)
        x = self.up(x)
        x = self.output(self.silu(self.group_norm(x)))
        return x


# Combines the Encoder and Decoder
# returns loss in forward() and has encode() and decode() methods
class AutoEncoderKL(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.encoder = Encoder(**config.model, image_size=config.dataset.image_size)
        self.decoder = Decoder(**config.model, image_size=config.dataset.image_size)
        self.kl_weight = config.model.get("kl_weight")
        
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        self.lpips.eval() 
        for param in self.lpips.parameters():
            param.requires_grad = False

        self.lpips_weight = config.model.get("lpips_weight")

        self.data_config = DataConfig(**config.dataset)

    def reparameterize(self, mu, log_var):
        # Reparameterization trick
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(mu)
        z = mu + std * eps
        return z

    def forward(self, x):
        # Encode
        mu, log_var = self.encode(x)

        # KL Loss
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1, 2, 3])
        )

        z = self.reparameterize(mu, log_var)

        # Decode
        x_hat = self.decode(z)

        # Reconstruction loss
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")

        # Perceptual loss
        # Compute with detached tensors
        with torch.no_grad():
            x_denorm = denormalize(self.data_config, x).detach()
            x_hat_denorm = denormalize(self.data_config, x_hat).detach()

        perceptual_loss = self.lpips(x_hat_denorm, x_denorm)

        total_loss = self.kl_weight * kl_loss + reconstruction_loss + self.lpips_weight * perceptual_loss

        return {
            "loss": total_loss,
            "kl_loss": kl_loss,
            "reconstruction_loss": reconstruction_loss,
            "perceptual_loss": perceptual_loss,
        }

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def sample(self, get_stats, device="cuda", sample_x=None, **kwargs):
        if sample_x is None:
            raise TypeError("sample_x must not be None")

        # Encode
        mu, log_var = self.encode(sample_x)

        z = self.reparameterize(mu, log_var)

        # Decode
        x = self.decode(z)

        # Denormalize images for viewing and saving
        return denormalize(self.data_config, x)
