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
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t=None):
        # Here, t already comes after SinusoidalEmbedding + a linear layer (and relu).
        # This time_mlp is used to change shape of the vector (pretty sure)
        h = self.act(self.norm1(self.conv1(x)))

        if self.time_mlp is not None and t is not None:
            time_emb = self.act(self.time_mlp(t))
            time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
            h += time_emb
            
        h = self.act(self.norm2(self.conv2(h)))
        h = self.dropout(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels*3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B,C,H,W = x.shape
        
        h = self.norm(x)
        q,k,v = self.qkv(h).chunk(3, dim=1)

        q = q.reshape(B,C,H*W)
        k = k.reshape(B,C,H*W)
        v = v.reshape(B,C,H*W)

        attn = torch.einsum("b c i, b c j -> b i j", q, k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)

        out = torch.einsum("b i j, b c j -> b c i", attn, v)
        out = out.reshape(B,C,H,W)

        return x + self.proj_out(out)
    
