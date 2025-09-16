import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10_000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        # self.relu = nn.ReLU()
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.residual_conv = (
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        # Here, t already comes after SinusoidalEmbedding + a linear layer (and relu).
        # This time_mlp is used to change shape of the vector (pretty sure)
        h = self.act(self.norm1(self.conv1(x)))
        time_emb = self.act(self.time_mlp(t))
        time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
        h += time_emb
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.residual_conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_multipliers: tuple,
        time_emb_dim: int = 64,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Input
        self.initial_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=base_channels, kernel_size=3, padding=1)
        
        channels = []
        for multiplier in channel_multipliers:
            channels.append(base_channels*multiplier)
        
        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(len(channel_multipliers))])

        forward_channels = [base_channels] + channels

        for i in range(len(forward_channels) - 1):
            self.downs.append(ResidualBlock(forward_channels[i], forward_channels[i+1], time_emb_dim))

        # Bottleneck
        # Doubling because it helps with keeping symmetry. The other option is to change the in_channels of self.ups elements
        bottleneck_in = channels[-1]
        bottleneck_out = bottleneck_in * 2
        self.bottleneck = ResidualBlock(bottleneck_in, bottleneck_out, time_emb_dim)

        # Decoder
        self.ups = nn.ModuleList()
        self.up_trans = nn.ModuleList()

        reversed_channels = [bottleneck_out] + list(reversed(channels))
        for i in range(len(reversed_channels) - 1):
            self.up_trans.append(nn.ConvTranspose2d(reversed_channels[i], reversed_channels[i+1], kernel_size=2, stride=2))
            # The input to the residual block will be doubled (after the skip connection is concatenated to the up_trans output)
            self.ups.append(ResidualBlock(reversed_channels[i], reversed_channels[i+1], time_emb_dim))

        # Output
        self.output = nn.Conv2d(base_channels, self.out_channels, kernel_size=1)
        
        # Zero-init final conv so initial predictions are near 0
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.initial_conv(x)

        skip_connections = []
        # Encoder
        for block, pool in zip(self.downs, self.pools):
            x = block(x, t_emb)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x, t_emb)

        # Decoder
        for block, up_transform, skip_connection in zip(self.ups, self.up_trans, reversed(skip_connections)):
            x = up_transform(x)
            x = torch.cat([x, skip_connection], dim=1)
            x = block(x, t_emb)
        
        return self.output(x)
    

if __name__ == "__main__":
    model = UNet(1, 1, 32, (1,2))
    print(model)