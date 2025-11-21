import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum
import math

from models.LFSR.det import ConvNet, SpatialTrans, FeatureAggregation
from models.HSLFSR.LFSR.DET.swin_angular import AngTrans

class CascadedAngTrans(nn.Module):
    def __init__(self, dim, num_heads, m):
        super().__init__()
        self.m = m
        self.transs = nn.ModuleList(
            [AngTrans(dim, num_heads, m, shifted=(i+1)%2==0) for i in range(2)]
        )

    def forward(self, x):
        for trans in self.transs:
            x = trans(x)
            x = rearrange(x, '(b h mh w mw) n c -> b c n (h mh) (w mw)', h=32//self.m, w=32//self.m, mh=self.m, mw=self.m)
        return x


class DET(nn.Module):
    def __init__(self, dim=64, num_heads=4, num_encoders=4, m=4, scale_factor=4):
        super().__init__()

        self.scale_factor = scale_factor

        ### shallow feature extraction
        self.in_conv = nn.Conv3d(1, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.conv = ConvNet(dim)

        ### deep feature extraction
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                SpatialTrans(dim, num_heads, s=2),
                CascadedAngTrans(dim, num_heads, m)
            ]) for _ in range(num_encoders)
        ])

        ### Hierarchical Feature Aggregation
        self.hfa = FeatureAggregation(dim, num_encoders=num_encoders)

        ### Upsampling
        self.up = nn.Sequential(
            nn.Conv2d(2*dim, scale_factor**2 * dim, kernel_size=1, bias=False),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        )

    def forward(self, x):
        
        #initial feature extraction
        buffer = self.in_conv(x.unsqueeze(1))
        buffer = buffer + self.conv(buffer)

        #dfe
        fs = []
        for spa, ang in self.encoders:
            buffer = ang(spa(buffer))
            fs.append(buffer)

        ###hfa
        buffer = self.hfa(fs)

        ### upsampling
        buffer = rearrange(buffer, 'b c n h w -> (b n) c h w')
        buffer = self.up(buffer)
        buffer = rearrange(buffer, '(b n) c h w -> (b c) n h w', n=25)

        return buffer + F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')