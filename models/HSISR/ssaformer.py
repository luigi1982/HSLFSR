import torch
from torch import nn
from torch.nn import functional as F
from dc1d.nn import DeformConv1d

from models.SISR.hat import OCAB

class SSAFormer(nn.Module):
    def __init__(self, channels, in_channels=25, num_groups=4, scale=4):
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.body = nn.Sequential(
            *[SSAG(channels=channels) for _ in range(num_groups)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
        self.up = nn.Sequential(
            nn.Conv2d(channels, scale**2 * channels, kernel_size=1, bias=False),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = x + self.body(x)
        return self.up(x)

class SSAG(nn.Module):
    def __init__(self, channels, num_blocks=4):
        super().__init__() 

        self.body = nn.Sequential(
            *[SSAB(channels=channels) for _ in range(num_blocks)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return x + self.body(x)

class SSAB(nn.Module):
    def __init__(self, channels):
        super().__init__()

        mlp_ratio = 2

        self.ln1 = nn.LayerNorm(channels)
        self.seb = SEB(channels=channels)
        self.sab = OCAB(channels=channels)
        self.ln2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_ratio*channels),
            nn.GELU(),
            nn.Linear(mlp_ratio*channels, channels)
        )

    def forward(self, x):
        x = x.permute((0, 2, 3, 1))
        buffer = self.ln1(x).permute((0, 3, 1, 2))
        x = self.sab(buffer).permute((0, 2, 3, 1)) + self.seb(buffer).permute((0, 2, 3, 1)) + x
        x = self.mlp(self.ln2(x)) + x
        return x.permute((0, 3, 1, 2))

class SEB(nn.Module):
    def __init__(self, channels, dconv_ks=3):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

        self.dconv1 = DeformConv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.dconv2 = DeformConv1d(1, 1, kernel_size=3, padding=1, bias=False)

        self.p1 = nn.Parameter(torch.zeros((1, channels, dconv_ks)))
        self.p2 = nn.Parameter(torch.zeros((1, channels, dconv_ks)))

    def forward(self, x):
        b = x.size(0)
        x = self.conv_net(x)
        attn = x.mean((2, 3)).unsqueeze(1)
        attn = self.dconv1(attn, self.p1.repeat(b, 1, 1, 1))
        attn = F.gelu(attn)
        attn = self.dconv2(attn, self.p2.repeat(b, 1, 1, 1))
        attn = F.sigmoid(attn)
        attn = attn.view((b, -1, 1, 1))
        return x * attn