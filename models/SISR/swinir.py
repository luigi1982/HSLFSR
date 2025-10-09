### Implementation of Swin Image Restoration (SwinIR)

import torch
from torch import nn
import math

from models.utils.swin import SwinBlock

class Upsample(nn.Sequential):
    def __init__(self, channels, scale):
        m = []

        #increase spatial dimension
        for _ in range(int(math.log(scale))):
            m.append(nn.Conv2d(channels, 4*channels, 3, 1, 1))
            m.append(nn.PixelShuffle(2))

        #reduce channel dimension to one
        m.append(nn.Conv2d(channels, out_channels=1, kernel_size=3, padding=1))

        super().__init__(*m)

    
class RSTB(nn.Module):
    def __init__(self, channels=96, num_layers=6):
        super().__init__()

        assert num_layers % 2 == 0, 'number of layers must be divisable by 2'

        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    SwinBlock(
                        hidden_dim=channels, heads=6, head_dim=16, mlp_dim=4*channels, shifted=False, window_size=8, relative_pos_embedding=None
                    ),
                    SwinBlock(
                        hidden_dim=channels, heads=6, head_dim=16 ,mlp_dim=4*channels, shifted=True, window_size=8, relative_pos_embedding=None
                    )
                ])
            )

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)


    def forward(self, x):

        #permute so it can be processed by the transformer layers
        x = x.permute(0, 2, 3, 1)
        for reg_stl, shifted_rsl in self.layers:
            x = reg_stl(x)
            x = shifted_rsl(x)

        #permute back, so it can be processed by conv
        x = x.permute(0, 3, 1, 2)

        x = self.conv(x)

        return x
    

class SwinIR(nn.Module):
    def __init__(self, embed_dim=96, num_blocks=4):
        super().__init__()

        num_feat = 64

        ### shallow feature extraction
        self.init_conv = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1)

        ### Deep Feature Extraction
        self.rstg = nn.Sequential(
            *[RSTB(channels=embed_dim) for _ in range(num_blocks)]
        )

        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        ### reconstruction
        self.conv_bfore_up = nn.Conv2d(embed_dim, num_feat, kernel_size=3, padding=1)
        self.up = Upsample(channels=num_feat, scale=4)

    def forward(self, x):
        x = self.init_conv(x)
        x = x + self.conv(self.rstg(x))
        x = self.up(self.conv_bfore_up(x))
        return x