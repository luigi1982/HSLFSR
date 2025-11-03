import torch
from torch import nn

from models.SISR.drcan import Upscaling, ResidualGroup

class DRCAN_concat(nn.Module):
    def __init__(self, channels=64, num_groups=10):
        super().__init__()

        #shallow feature extraction
        self.sfe = nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)
        #deep feature extraction
        self.dfe = nn.Sequential(
            *[ResidualGroup(channels=channels) for _ in range(num_groups)]
        )
        #upscaling
        self.up = Upscaling(channels=channels, scale_factor=4)

    def forward(self, x, wl):

        #normalize the wavelength
        wl = wl/24
        #concatenate wavelength and lr image
        x = torch.cat([x, wl[:, :, None, None]*torch.ones_like(x)], dim=1)

        #pass through the model
        buffer = self.sfe(x)
        x = buffer + self.dfe(buffer)
        x = self.up(x)
        return x