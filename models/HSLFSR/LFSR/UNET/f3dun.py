import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from models.utils.commons import Mish, SinusoidalPosEmb

class F3DUN_UNET(nn.Module):
    def __init__(self, dim, time_emb_dim, cond_dim, scale_factor=4):
        super().__init__()
        self.init_conv = nn.Conv3d(1, dim, 3, 1, 1)
        self.cond_proj = nn.ConvTranspose3d(
            cond_dim,
            dim,
            (1, scale_factor * 2, scale_factor * 2),
            (1, scale_factor, scale_factor),
            (0, scale_factor // 2, scale_factor // 2),
        )
        self.unet = F3DUN(dim, time_emb_dim)
        self.conv_out = nn.Conv3d(dim, 1, 3, 1, 1)

    def forward(self, x, cond, t):
        cond = self.cond_proj(cond)
        x = self.init_conv(x) + cond
        x = self.unet(x, t)
        x = self.conv_out(x)
        x = rearrange(x, 'b c n h w -> (b c) n h w')
        return x
    

class ResBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv3d(dim, dim, 3, 1, 1)

        #time embedding
        self.emb = nn.Sequential(
            Mish(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, t):
        buffer = self.conv1(x) 
        buffer = F.relu(buffer)
        buffer = buffer + self.emb(t)[:, :, None, None, None]
        buffer = self.conv2(buffer)
        return x + buffer


class F3DUN(nn.Module):
    def __init__(self, channels=64, time_emb_dim=64):
        super().__init__()
        dim = channels
        self.emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            Mish(),
            nn.Linear(time_emb_dim, dim)
        )

        self.shallow = nn.ModuleList(
            [ResBlock(dim=dim) for _ in range(5)]
        )
        self.deep = nn.ModuleList([
            nn.ModuleList([
                nn.Conv3d(2*dim, dim, 1), ResBlock(dim=dim)
            ]) for _ in range(5)
        ])

    def forward(self, x, t):

        #embed time
        t = self.emb(t)

        buffer1 = x

        #shallow half
        h = [] #save skip connections
        buffer2 = buffer1
        for i, res in enumerate(self.shallow):
            buffer2 = res(buffer2, t)
            #add skip connection
            if i == 4:
                buffer2 = buffer2 + buffer1
            h.append(buffer2)

        buffer1 = buffer2

        #deep half
        for fuse, res in self.deep:
            buffer2 = torch.concat([buffer2, h.pop()], dim=1)
            buffer2 = fuse(buffer2)
            buffer2 = res(buffer2, t)

        #add skip connection
        buffer2 = buffer2 + buffer1

        #sample up
        return buffer2