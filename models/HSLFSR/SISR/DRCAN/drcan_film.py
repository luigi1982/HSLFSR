import torch
from torch import nn

from models.SISR.drcan import ChannelAttention, Upscaling


class FiLM(nn.Module):
    def __init__(self, channels=64, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 2*channels)
        )

    def forward(self, wl):
        gamma, beta = self.mlp(wl).chunk(2, dim=-1)
        return gamma[:, :, None, None], beta[:, :, None, None]
    

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

    def forward(self, x, gamma, beta):
        return x + gamma * self.conv_net(x) + beta
    
class ResidualGroup(nn.Module):
    def __init__(self, channels=64, num_blocks=20):
        super().__init__()
        self.blocks = nn.ModuleList([
            RCAB(channels) for _ in range(num_blocks)
        ])

        self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, gamma, beta):
        buffer = x
        for block in self.blocks:
            buffer = block(buffer, gamma, beta)

        buffer = self.out_conv(buffer)

        return x + buffer

class DRCAN_FiLM(nn.Module):

    def __init__(self, channels=64, num_groups=10):
        super().__init__()

        #FiLM
        self.film = FiLM(channels=channels)
        #shallow feature extraction
        self.sfe = nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False)
        #deep feature extraction
        self.groups = nn.ModuleList(
            [ResidualGroup(channels=channels) for _ in range(num_groups)]
        )
        #upscaling
        self.up = Upscaling(channels=channels, scale_factor=4)

    def forward(self, x, wl):

        ### normalize wavelength
        wl = wl/24
        ### get gamma and beta from FiLM network
        gamma, beta = self.film(wl)
        
        ### pass through network
        x = gamma * self.sfe(x) + beta
        buffer = x
        for group in self.groups:
            buffer = group(buffer, gamma, beta)

        x = x + buffer

        x = self.up(x)
        
        return x