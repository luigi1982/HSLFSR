import torch
from torch import nn

class PixelShuffle1d(nn.Module):
    def __init__(self, sf):
        super().__init__()
        self.sf = sf

    def forward(self, x):
        b, sfc, h, w = x.shape
        c = sfc//self.sf
        x = x.view((b, c, self.sf, h, w))
        x = x.permute((0, 1, 3, 2, 4)).contiguous().view((b, c, h, self.sf*w)) 
        return x
    
class SFEConv(nn.Module):
    def __init__(self, dim, A, dim_out=None):
        super().__init__()
        if not dim_out:
            dim_out = dim
        self.sfe_conv = nn.Conv2d(
                dim, dim_out, 3, stride=1, dilation=A, padding=A
        )

    def forward(self, x):
        return self.sfe_conv(x)

class SFE(nn.Module):
    def __init__(self, dim, A):
        super().__init__()
        self.net = nn.Sequential(
            SFEConv(dim, A),
            nn.LeakyReLU(0.1),
            SFEConv(dim, A)
        )

    def forward(self, x):
        return self.net(x)
    
class AFE(nn.Module):
    def __init__(self, dim, dim_out, A):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out, A, stride=A),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, A**2*dim_out, 1),
            nn.PixelShuffle(A)
        )

    def forward(self, x):
        return self.net(x)
    
class EFE(nn.Module):
    def __init__(self, dim, dim_out, A):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out, (1, A), stride=(1, A)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, A*dim_out, 1),
            PixelShuffle1d(A)
        )

    def forward(self, x):
        return self.net(x)
    

class DistgBlock(nn.Module):
    def __init__(self, dim, A, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        sfe_dim, afe_dim, efe_dim = dim, dim//4, dim//2
        self.sfe = SFE(dim, A)
        self.afe = AFE(dim, afe_dim, A)
        self.efe = EFE(dim, efe_dim, A)

        self.fuse = nn.Sequential(
            nn.Conv2d(sfe_dim + afe_dim + 2*efe_dim, dim_out, 1),
            nn.LeakyReLU(0.1),
            SFEConv(dim_out, A)
        )

    def forward(self, x, use_res=True):
        x_s = self.sfe(x)
        x_a = self.afe(x)
        x_h = self.efe(x)
        x_v = self.efe(x.permute((0, 1, 3, 2)))

        buffer = torch.concat([x_s, x_a, x_h, x_v], dim=1)
        buffer = self.fuse(buffer)

        if use_res:
            return x + buffer
        else:
            return buffer
    
class DistgGroup(nn.Module):
    def __init__(self, dim, A):
        super().__init__()
        self.blocks = nn.Sequential(
            DistgBlock(dim, A),
            DistgBlock(dim, A),
            DistgBlock(dim, A),
            DistgBlock(dim, A)
        )

    def forward(self, x):
        return x + self.blocks(x)
    
class DISTG(nn.Module):
    def __init__(self, channels, A=5):
        super().__init__()
        self.dim = channels
        self.init_conv = SFEConv(1, A, dim_out=channels)
        self.groups = nn.Sequential(
            DistgGroup(channels, A),
            DistgGroup(channels, A),
            DistgGroup(channels, A),
            DistgGroup(channels, A),
        )
        self.net_out = nn.Sequential(
            nn.Conv2d(channels, 16, 1),
            nn.PixelShuffle(4)
        )

    def forward(self, x, A=5):

        b, _, h, w = x.shape
        u, v = A, A

        #reshape to MacPi
        buffer = x.view((b, u, v, h, w))
        buffer = buffer.permute((0, 3, 4, 1, 2)).contiguous().view((b, 1, u*h, v*w))
       
        #pass x through net
        buffer = self.init_conv(buffer)
        buffer = buffer + self.groups(buffer)

        #reshape to SAI and sample up 
        buffer = buffer.view((b, self.dim, h, w, u, v)).permute((0, 4, 5, 1, 2, 3))
        buffer = buffer.contiguous().view((b*u*v, self.dim, h, w))
        buffer = self.net_out(buffer)
        x = x.view((b, u*v, h, w))
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear')
        x = x.view((b, u*v, 4*h, 4*w))
        buffer = buffer.view((b, u*v, 4*h, 4*w))

        return buffer + x
