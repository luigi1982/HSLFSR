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

        #define blocks    
        self.block1 = DistgBlock(dim, A, dim_out=dim_out)
        self.block2 = DistgBlock(dim_out, A)

        #residual connection
        self.res_connect = SFEConv(dim, A, dim_out=dim_out)

    def forward(self, x):
        buffer = self.block1(x, use_res=False)
        buffer = self.block2(buffer, use_res=False)
        return buffer + self.res_connect(x)
    

class UNET(nn.Module):
    def __init__(self, dim, use_attention=False):
        super().__init__()
        
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

    def forward(self, x):

        #skip connections
        h = []
        
        #go through contracting path
        for res1, res2, down in self.down:
            x = res1(x)
            x = res2(x)
            h.append(x)
            x = down(x)

        #go through bottleneck
        x = self.mid1(x)
        x = self.attention(x)
        x = self.mid2(x)

        #go through expanding path
        for res1, res2, up in self.up:
            x = torch.concat([x, h.pop()], dim=1)
            x = res1(x)
            x = res2(x)
            x = up(x)

        return x
    
class DISTG_UNET_regression(nn.Module):
    def __init__(self, dim):
        super().__init__()

        A=5

        self.init_conv = SFEConv(1, A, dim_out=dim)
        self.unet = UNET(dim)
        self.out_conv = SFEConv(dim, A)

        self.upsampling = nn.Sequential(
            nn.Conv2d(dim, dim*16, 1, bias=False),
            nn.PixelShuffle(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 1, 3, padding=1, bias=False)
        )

    def forward(self, x):

        [_, _, h, w] = x.size()

        x = rearrange(x, 'b (u v) h w -> b (h u) (w v)', u=5, v=5)
        x = x.unsqueeze(1)
        
        x = self.init_conv(x)
        x = self.unet(x)
        x = self.out_conv(x)

        x_up = rearrange(x, 'b c (h u) (w v) -> (b u v) c h w', u=5, v=5)
        x_up = self.upsampling(x_up)
        x_up = rearrange(x_up, '(b u v) c h w -> (b c) (u v) h w', h=4*h, w=4*w, u=5, v=5)

        return x_up