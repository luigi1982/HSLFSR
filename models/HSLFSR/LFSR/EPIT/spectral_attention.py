import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum

import math

from models.LFSR.epit import Residual, SpatialConv

class SpactralAttention(nn.Module):

    def __init__(self, emb_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = math.sqrt(emb_dim)

        self.to_qk = nn.Linear(emb_dim, 2*emb_dim)
        self.wl_proj = nn.Parameter(
            torch.randn((5, emb_dim//num_heads, emb_dim//num_heads))
        )

    def forward(self, x):

        q, k, v = map(
            lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads),
            [*(self.to_qk(x).chunk(2, dim=-1))] + [x]
        )

        attn = einsum(q, k, 'b h l1 d, b h l2 d -> b h l1 l2')/self.scale
        attn = F.softmax(attn, dim=-1)
        v = rearrange(v, 'b h (v w) d -> b h v w d', v=5)
        out = v + einsum(self.wl_proj, v, 'v d1 d2, b h v w d1 -> b h v w d2')
        out = rearrange(out, 'b h v w d -> b h (v w) d')
        out = einsum(attn, out, 'b h l1 l2, b h l2 d -> b h l1 d')
        out = rearrange(out, 'b h l d -> b l (h d)')

        return out
    
class BasicTrans(nn.Module):
    def __init__(self, dim, emb_dim):
        super().__init__()
        self.in_ = nn.Linear(dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.mhsa = SpactralAttention(emb_dim)
        self.out = nn.Linear(emb_dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )
    def forward(self, x):

        x = rearrange(x, 'b c (u v) h w -> (b u h) (v w) c', u=5)

        x = self.in_(x)
        buffer = self.ln(x)
        x = buffer + self.mhsa(buffer)
        x = x + self.mlp(x)
        x = self.out(x)

        x = rearrange(x, '(b u h) (v w) c -> b c (u v) h w', u=5, v=5, h=32)

        return x
    
class AltFilter(nn.Module):
    def __init__(self, channels, ang_res=5):
        super().__init__()
        self.ang_res = ang_res
        self.epi_trans = BasicTrans(channels, 2*channels)
        self.conv = SpatialConv(channels)

    def forward(self, x):

        shortcut = x
        [_, _, _, h, w] = x.size()

        #extract horizontal EPI features
        x = self.epi_trans(x)
        x = self.conv(x) + shortcut

        #extract vertical EPI features
        x = rearrange(x, 'b c (u v) h w -> b c (v u) w h', u=self.ang_res, v=self.ang_res)
        x = self.epi_trans(x)
        x = rearrange(x, 'b c (v u) w h -> b c (u v) h w', u=self.ang_res, v=self.ang_res, h=h, w=w)
        x = self.conv(x) + shortcut

        return x
    

class EPIT(nn.Module):
    def __init__(self, channels, ang_res=5, use_as_encoder=False):
        super().__init__()

        self.ang_res = ang_res
        self.use_as_encoder = use_as_encoder

        self.in_conv = nn.Sequential(
            nn.Conv3d(1, channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            Residual(SpatialConv(channels))
        )

        self.EPI_features = nn.Sequential(
            *[AltFilter(channels) for _ in range(5)]
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels*16, 1, bias=False),
            nn.PixelShuffle(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, 3, padding=1, bias=False)
        )

    def forward(self, x):
        [_, _, h, w] = x.size()
        x = self.in_conv(x.unsqueeze(1))
        x = self.EPI_features(x) + x

        # x is shape B x C x UV x H x W
        # reshape to B x C x UV x HW for upsampling
        x_up = rearrange(x, 'b c (u v) h w -> b c (u h) (v w)', u=self.ang_res, v=self.ang_res)
        x_up = self.upsampling(x_up)
        x_up = rearrange(x_up, 'b c (u h) (v w) -> b c (u v) h w', h=4*h, w=4*w, u=self.ang_res, v=self.ang_res)

        if self.use_as_encoder:
            return x_up, rearrange(x, 'b c (u v) h w -> b c (h u) (w v)', u=self.ang_res, v=self.ang_res)
        else:
            x_up = x_up.view((-1, 25, 128, 128))
            return x_up