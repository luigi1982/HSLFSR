import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.conv(x)
    
class F3DUN(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.init_conv = nn.Conv3d(1, dim, 3, 1, 1)

        self.shallow = nn.ModuleList(
            [ResBlock(dim=dim) for _ in range(5)]
        )
        self.deep = nn.ModuleList(
            [nn.Sequential(nn.Conv3d(2*dim, dim, 1), ResBlock(dim=dim)) for _ in range(5)]
        )

        #upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 4, 4)),
            nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(dim, 1, (3, 1, 1), padding=(1, 0, 0)),
        )

    def forward(self, x):

        #initial convolution
        buffer1 = self.init_conv(x.unsqueeze(1))

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

        #sample up
        x = F.interpolate(x, scale_factor=4, mode='nearest')

        buffer2 = self.up(buffer2)
        b, _, n, h, w = buffer2.shape
        buffer2 = buffer2.view((b, n, h, w))
        return x + buffer2



        