import torch
from torch import nn
from torchvision.ops import DeformConv2d
from torch.nn import functional as F

class ASPPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv_dila1 = nn.Conv2d(
            dim, dim, 3, stride=1, padding=1, dilation=1
        )
        self.conv_dila2 = nn.Conv2d(
            dim, dim, 3, stride=1, padding=2, dilation=2
        )
        self.conv_dila4 = nn.Conv2d(
            dim, dim, 3, stride=1, padding=4, dilation=4
        )

        self.fuse = nn.Conv2d(
            3*dim, dim, 1
        )

    def forward(self, x):
        d1 = self.conv_dila1(x)
        d2 = self.conv_dila2(x)
        d4 = self.conv_dila4(x)

        return x + self.fuse(torch.concat([d1, d2, d4], dim=1))
        

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, x):
        return x + self.conv_net(x)
    
class FeatureExtraction(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.init_conv = nn.Conv2d(1, dim, 1)
        self.res_group = nn.Sequential(
            ASPPBlock(dim),
            ResBlock(dim),
            ASPPBlock(dim),
            ResBlock(dim),
        )

    def forward(self, x):

        #expects inputs
        # x to be B x 1 x H x W

        x = self.init_conv(x)
        x = self.res_group(x)
        return x
    
class ADAM(nn.Module):
    def __init__(self, dim, A=5):
        super().__init__()

        #offset generation branch
        self.offset_branch = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.LeakyReLU(0.1),
            ASPPBlock(dim),
            nn.Conv2d(dim, 18, 1)
        )

        #deformable convolution
        self.dconv = DeformConv2d(dim, dim, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1)

        #fusing networks
        self.fuse1 = nn.Conv2d(A**2*dim, A**2*dim, 1)
        self.fuse2 = nn.Conv2d(2*dim, dim, 1)

    def forward(self, x_sv, x_cv):

        b, n, c, h, w = x_sv.shape

        # x_sv contains side views - B x N x C x H x W
        # s_cv central view - B x C x H x W

        #collect
        #compute offsets for deformable convolution
        aligned = []
        for i in range(n):
            x = x_sv[:, i, :, :, :]
            xx_cv = torch.concat([x, x_cv], dim=1) #concatenate with central view
            o = self.offset_branch(xx_cv) #compute offsets
            aligned.append(self.lrelu(self.dconv(x, o)))

        aligned = torch.concat(aligned+[x_cv], dim=1)   
        #fuse features
        fused = self.fuse1(aligned).unsqueeze(1).view((b, -1, c, h, w))

        #distribute
        out_sv = []
        for i in range(n):
            x = x_sv[:, i, :, :, :]
            f = fused[:, i+1, :, :, :]
            xf = torch.concat([x, f], dim=1)
            o = self.offset_branch(xf)
            x = self.fuse2(torch.concat([x, self.lrelu(self.dconv(x, o))], dim=1))
            out_sv.append(x)

        x_cv = self.fuse2(torch.concat([x_cv, fused[:, 0, :, :, :]], dim=1))
        out_sv = torch.concat(out_sv, dim=1).view((b, -1, c, h, w))

        return out_sv, x_cv

# the Information Module Distillation Block    
class IMDB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        #initial convolition
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        #mapping wide features back to original channel dimension
        #in conv2 and conv3
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*dim//4, dim, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3*dim//4, dim, 3, padding=1),
            nn.LeakyReLU(0.1)
        )

        # mapping the last wide features to the same dimension as the narrow features
        self.conv4 = nn.Sequential(
            nn.Conv2d(3*dim//4, dim//4, 3, padding=1),
            nn.LeakyReLU(0.1)
        )

        #fusing features
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        c = x.size(1)
        buffer = self.conv1(x)
        n1, buffer = torch.tensor_split(buffer, (c//4,), dim=1)
        buffer = self.conv2(buffer)
        n2, buffer = torch.tensor_split(buffer, (c//4,), dim=1)
        buffer = self.conv3(buffer)
        n3, buffer = torch.tensor_split(buffer, (c//4,), dim=1)
        n4 = self.conv4(buffer)

        #fuse narrow features and add residual connection
        return x + self.fuse(torch.concat([n1, n2, n3, n4], dim=1))
    
class RecModule(nn.Module):
    def __init__(self, dim, num_blocks=3):
        super().__init__()
        self.net = nn.Sequential(
            *[IMDB(dim) for _ in range(num_blocks)]
        )
        
    def forward(self, x):
        return self.net(x)
    
class Upsampling(nn.Module):
    def __init__(self, dim, factor=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4*dim, factor**2*dim, 1),
            nn.PixelShuffle(factor),
            nn.Conv2d(dim, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
    

class NET(nn.Module):
    def __init__(self, channels):
        super().__init__()
        dim = channels
        self.feature_extraction = FeatureExtraction(dim)
        self.adams = nn.ModuleList(
            [ADAM(dim) for _ in range(3)]
        )
        self.reconstruction = RecModule(4*dim)
        self.up = Upsampling(dim)

    def forward(self, x):

        #x is expected to be B x UV x H x W
        b, n, h, w = x.shape

        #network
        buffer = self.feature_extraction(x.contiguous().view((b*n, 1, h, w)))
        buffer = buffer.view((b, n, 32, h, w))
        b_sv, b_cv = LF_Split(buffer)

        buffer_sv, buffer_cv = [b_sv], [b_cv]
        for adam in self.adams:
            b_sv, b_cv = adam(b_sv, b_cv)
            buffer_sv.append(b_sv)
            buffer_cv.append(b_cv)

        buffer_sv = torch.concat(buffer_sv, dim=2)
        buffer_sv = buffer_sv.view((b*(n-1), -1, h, w))
        buffer_cv = torch.concat(buffer_cv, dim=1)

        buffer_sv = self.reconstruction(buffer_sv)
        buffer_sv = self.up(buffer_sv)
        buffer_sv = buffer_sv.view((b, n-1, 4*h, 4*w))

        buffer_cv = self.reconstruction(buffer_cv)
        buffer_cv = self.up(buffer_cv)

        buffer = LF_Fuse(buffer_sv, buffer_cv)

        #bicubic upsampling
        x = F.interpolate(x, scale_factor=4, mode='bicubic')

        return x + buffer

def LF_Split(x, A=5):
    i = A**2//2
    return torch.concat([x[:, :i], x[:, i+1:]], dim=1), x[:, i]

def LF_Fuse(x_sv, x_cv, A=5):
    i = (A**2 - 1) // 2
    return torch.concat([x_sv[:, :i], x_cv, x_sv[:, i:]], dim=1)