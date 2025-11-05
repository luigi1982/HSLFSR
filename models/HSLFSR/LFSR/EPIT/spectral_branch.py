import torch
from torch import nn
from einops import rearrange

from models.LFSR.epit import SpatialConv, BasicTrans, Residual
from models.SSR.mstpp import SMSA


### Residual Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels//r),
            nn.ReLU(),
            nn.Linear(channels//r, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ### Input is going to be B x C x UV x H x W
        x = x.transpose(1, 2)
        buffer = x.mean((3, 4))
        buffer = self.attn(buffer)
        x = x * buffer[:, :, :, None, None]
        return x.transpose(1, 2)

class RCAB(nn.Sequential):
    def __init__(self, channels):

        ### conv -> ReLU -> conv -> CA -> res
        super().__init__(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            ChannelAttention(channels)
        )


### MST++
class SABranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.smsa = SMSA(25, 5)
        self.conv_in = nn.Conv3d(channels, 1, kernel_size=1)
        self.conv_out = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        b, c, n, h, w = x.size()
        x = self.conv_in(x)
        x = x.view((b, n, h, w)).permute((0, 2, 3, 1)).view((b, h*w, n))
        x = self.smsa(x)
        x = x.view((b, h, w, n)).permute((0, 3, 1, 2)).view((b, 1, n, h, w))
        x = self.conv_out(x)
        x = self.ca(x)
        return x

class AltFilter(nn.Module):
    def __init__(self, channels, ang_res=5, spectral_branch=RCAB):
        super().__init__()
        self.ang_res = ang_res
        self.epi_trans = BasicTrans(channels, 2*channels)
        self.spectral_branch = spectral_branch(channels)
        self.conv = SpatialConv(channels)

    def forward(self, x):
        
        '''
        Input is stacked SAI, B x UV x H x W x C
        For horizontal EPI feature extraction reshape to VW x BHU x C 
        (VW is first as batch_first=False is standard)
        For SpatialConv back to B x UV x H x W x C
        For vertical EPI feature extraction to UH x BVW x C
        For SpatialConv back to B x UV x H x W x C

        Residual connections are added after both SpatialConvs
        '''

        shortcut = x
        [_, _, _, h, w] = x.size()

        #extract horizontal EPI features
        buffer = rearrange(x, 'b c (u v) h w -> (v w) (b u h) c', u=self.ang_res, v=self.ang_res)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, '(v w) (b u h) c -> b c (u v) h w', u=self.ang_res, v=self.ang_res, h=h, w=w)
        x = buffer + self.spectral_branch(x)
        x = self.conv(x) + shortcut

        #extract vertical EPI features
        buffer = rearrange(x, 'b c (u v) h w -> (u h) (b v w) c', u=self.ang_res, v=self.ang_res)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, '(u h) (b v w) c -> b c (u v) h w', u=self.ang_res, v=self.ang_res, h=h, w=w)
        x = x + self.spectral_branch(x)
        x = self.conv(x) + shortcut

        return x
    

class EPIT(nn.Module):
    def __init__(self, channels, ang_res=5, use_as_encoder=False, spectral_branch=RCAB):
        super().__init__()

        self.ang_res = ang_res
        self.use_as_encoder = use_as_encoder

        self.in_conv = nn.Sequential(
            nn.Conv3d(1, channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            Residual(SpatialConv(channels))
        )

        self.EPI_features = nn.Sequential(
            *[AltFilter(channels, spectral_branch=spectral_branch) for _ in range(5)]
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