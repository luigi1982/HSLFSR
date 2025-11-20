import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from models.utils.conv4d import Conv4d

class ResBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            Conv4d(dim, dim, 3, padding=(0, 0, 1, 1)),
            nn.ReLU(),
            Conv4d(dim, dim, 3, padding=(0, 0, 1, 1))
        )

    def forward(self, x):
        print(x.shape)
        return x + self.conv(x)
    
class F4DUN(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        dim = channels
        self.init_conv = Conv4d(1, dim, 3, padding=(1, 1, 1, 1))

        self.shallow = nn.ModuleList(
            [ResBlock(dim=dim) for _ in range(5)]
        )
        self.deep = nn.ModuleList(
            [nn.Sequential(Conv4d(2*dim, dim, 1), ResBlock(dim=dim)) for _ in range(5)]
        )

        #upsampling
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels*16, 1, bias=False),
            nn.PixelShuffle(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, 3, padding=1, bias=False)
        )

    def forward(self, x):

        #initial convolution
        buffer1 = rearrange(x, 'b (c u v) h w -> b c u v h w', c=1, u=5)

        print(buffer1.shape)
        buffer1 = self.init_conv(buffer1)
        print(buffer1.shape)

        print('initial conv over')

        #shallow half
        h = [] #save skip connections
        buffer2 = buffer1
        for i, res in enumerate(self.shallow):
            buffer2 = res(buffer2)
            #add skip connection
            if i == 4:
                buffer2 += buffer1
            h.append(buffer2)

        buffer1 = buffer2

        #deep half
        for res in self.deep:
            buffer2 = torch.concat([buffer2, h.pop()], dim=1)
            buffer2 = res(buffer2)

        #add skip connection
        buffer2 += buffer1

        #use nn for upsampling
        x = F.interpolate(x, scale_factor=4, mode='nearest')

        # sample up the features
        # they are of shape B x C x U x V x H x W
        # reshape to (B U V) x C x H x W
        # upsample each SAI individually

        buffer2 = rearrange(buffer2, 'b c u v h w -> (b u v) c h w')
        buffer2 = self.up(buffer2)
        buffer2 = rearrange(buffer2, '(b u v) c h w -> b (c u v) h w', u=5, v=5)
        return x + buffer2



        