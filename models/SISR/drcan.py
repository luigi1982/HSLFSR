### Deep Residual Channel Attention Network

import torch
from torch import nn

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
        return x * self.attn(x.mean((2, 3)))[:, :, None, None]
    
class RCAB(nn.Module):
    ### Residual Channel Attention Block 

    def __init__(self, channels):
        super().__init__()

        ### conv -> ReLU -> conv -> CA -> res
        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            ChannelAttention(channels)
        )

    def forward(self, x):
        return x + self.conv_net(x)
    
class ResidualGroup(nn.Module):
    def __init__(self, channels=64, num_blocks=20):
        super().__init__()
        self.conv_net = nn.Sequential(
            *[RCAB(channels) for _ in range(num_blocks)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.conv_net(x)
    
class Upscaling(nn.Module):
    def __init__(self, channels=64, scale_factor=4):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, scale_factor**2 * channels, kernel_size=1, bias=False),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.conv_net(x)
    
class DRCAN(nn.Module):
    def __init__(self, channels=64, num_groups=10):
        super().__init__()

        #shallow feature extraction
        self.sfe = nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False)
        #deep feature extraction
        self.dfe = nn.Sequential(
            *[ResidualGroup(channels=channels) for _ in range(num_groups)]
        )
        #upscaling
        self.up = Upscaling(channels=channels, scale_factor=4)

    def forward(self, x):
        buffer = self.sfe(x)
        x = buffer + self.dfe(buffer)
        x = self.up(x)
        return x