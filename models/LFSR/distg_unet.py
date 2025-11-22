import torch
from torch import nn
from einops import rearrange
import math

from models.LFSR.distg import DistgBlock, SFEConv
from models.utils.commons import SinusoidalPosEmb, Mish, Downsample, Upsample, LinearAttention

class DistgResGroup(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()

        A=5

        if dim_out is None:
            dim_out=dim

        #time embedding
        self.emb = nn.Sequential(
            Mish(),
            nn.Linear(dim_out, dim_out)
        )

        #define blocks    
        self.block1 = DistgBlock(dim, A, dim_out=dim_out)
        self.block2 = DistgBlock(dim_out, A)

        #residual connection
        self.res_connect = SFEConv(dim, A, dim_out=dim_out)

    def forward(self, x, t):
        t = self.emb(t)
        buffer = self.block1(x, use_res=False)
        buffer = self.block2(buffer + t[:, :, None, None], use_res=False)
        return buffer + self.res_connect(x)
    
class UNET(nn.Module):
    def __init__(self, dim, time_emb_dim, use_attention=False):
        super().__init__()

        #time embedding
        self.emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            Mish(),
            nn.Linear(time_emb_dim, dim)
        )
        
        #contracting and expanding path
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])

        #populate contracting and expanding path
        for i in range(3):

            #contracting path
            self.down.append(
                nn.ModuleList([
                    DistgResGroup(dim),
                    DistgResGroup(dim),
                    Downsample(dim) if i < 2 else nn.Identity()
                ])
            )

            #expanding path
            self.up.append(
                nn.ModuleList([
                    DistgResGroup(2*dim, dim_out=dim),
                    DistgResGroup(dim, dim),
                    Upsample(dim) if i < 2 else nn.Identity()
                ])
            )

        #define the bottleneck
        self.mid1 = DistgResGroup(dim, dim)
        self.mid2 = DistgResGroup(dim, dim)
        self.attention = LinearAttention(dim) if use_attention else nn.Identity()

    def forward(self, x, t):

        #embed time
        t = self.emb(t)
        #skip connections
        h = []
        
        #go through contracting path
        for res1, res2, down in self.down:
            x = res1(x, t)
            x = res2(x, t)
            h.append(x)
            x = down(x)

        #go through bottleneck
        x = self.mid1(x, t)
        x = self.attention(x)
        x = self.mid2(x, t)

        #go through expanding path
        for res1, res2, up in self.up:
            x = torch.concat([x, h.pop()], dim=1)
            x = res1(x, t)
            x = res2(x, t)
            x = up(x)

        return x
    
class DISTG_UNET(nn.Module):
    def __init__(self, dim, time_emb_dim, cond_dim, scale_factor=4):
        super().__init__()

        A=5

        self.cond_proj = nn.ConvTranspose2d(
            cond_dim,
            dim,
            scale_factor * 2,
            scale_factor,
            scale_factor // 2,
        )
        self.init_conv = SFEConv(1, A, dim_out=dim)
        self.unet = UNET(dim, time_emb_dim)
        self.out_conv = SFEConv(dim, A, dim_out=1)

    def forward(self, x, cond, t):

        #cond and x are expected to be MacPis
        x = rearrange(x, 'b c (u v) h w -> b c (h u) (w v)', u=5, v=5)
        cond = rearrange(cond, 'b c (u v) h w -> b c (h u) (w v)', u=5, v=5)
        
        cond = self.cond_proj(cond)
        x = self.init_conv(x) + cond
        x = self.unet(x, t)
        x = self.out_conv(x)

        return x
        


        
            

        


