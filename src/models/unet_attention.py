import torch
import torch.nn as nn

from .blocks import SinusoidalPositionEmbeddings, ResidualBlock, AttentionBlock


class UNetWithAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_multipliers: tuple,
        attention_resolutions: tuple,
        time_emb_dim: int = 64,
        image_size: int = 32,  # for calculating resolutions for attention block placement
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

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
        self.downsamplers = nn.ModuleList()

        forward_channels = [base_channels] + channels
        current_res = self.image_size

        for i in range(len(forward_channels) - 1):
            in_ch = forward_channels[i]
            out_ch = forward_channels[i+1]

            # Add residual blocks
            self.downs.append(ResidualBlock(in_ch, out_ch, time_emb_dim))

            # Check and add attention blocks
            if current_res in attention_resolutions:
                self.downs.append(AttentionBlock(out_ch))

            self.downsamplers.append(nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1))
            
            current_res //= 2  # Since we always half the resolution in the pools
            

        # Bottleneck
        # Doubling because it helps with keeping symmetry. The other option is to change the in_channels of self.ups elements
        bottleneck_in = channels[-1]
        bottleneck_out = bottleneck_in * 2
        # The bottleneck adds an Attention layer by default. A little different from the paper, but it should only help
        # I could try and removing it later on to see if there's any change. It can't really make it better (except improved training and inference tiems I suppose)
        self.bottleneck = nn.ModuleList([
            ResidualBlock(bottleneck_in, bottleneck_out, time_emb_dim),
            AttentionBlock(bottleneck_out),
            ResidualBlock(bottleneck_out, bottleneck_out, time_emb_dim),
        ])

        # Decoder
        self.ups = nn.ModuleList()
        # self.up_trans = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        reversed_channels = [bottleneck_out] + list(reversed(channels))
        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i+1]

            self.upsamplers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
                )
            )

            # The input to the residual block will be doubled (after the skip connection is concatenated to the up_trans output)
            self.ups.append(ResidualBlock(in_ch, out_ch, time_emb_dim))

            # Check and add attention block
            current_res *= 2
            if current_res in attention_resolutions:
                self.ups.append(AttentionBlock(out_ch))


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

        # Encoder with attention
        down_idx = 0
        for i in range(len(self.downsamplers)):
            # Residual block for this level
            x = self.downs[down_idx](x, t_emb)
            down_idx += 1

            # Check if there's an Attention Block at this level
            if down_idx < len(self.downs) and isinstance(self.downs[down_idx], AttentionBlock):
                x = self.downs[down_idx](x)
                down_idx += 1

            # Save the skip connection
            skip_connections.append(x)

            # Down sampling
            x = self.downsamplers[i](x)

        # Bottleneck is effectively static
        x = self.bottleneck[0](x, t_emb)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t_emb)

        # Decoder with attention
        up_idx = 0
        for i in range(len(self.upsamplers)):
            # Get the skip connection
            skip = skip_connections.pop()

            # Upsampling
            x = self.upsamplers[i](x)

            # Add skip connection (concatenate)
            x = torch.cat([x, skip], dim=1)

            # Residual block
            x = self.ups[up_idx](x, t_emb)
            up_idx += 1

            # Check and use attention block
            if up_idx < len(self.ups) and isinstance(self.ups[up_idx], AttentionBlock):
                x = self.ups[up_idx](x)
                up_idx += 1        
        
        return self.output(x)