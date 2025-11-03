import torch
from torch import nn
from torch.nn import functional as F
from einops import einsum, rearrange
import math

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        # input is expected to be of shape B x L x D
        # where L composed of H x W
        # reshape to B x D x H x W
        l = x.size(1)
        h = int(math.sqrt(l))
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=h)
        x = self.conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x

class MLP(nn.Sequential):
    def __init__(self, dim_in, dim_hidden=None, dim_out=None):
        dim_hidden = dim_hidden if dim_hidden else dim_in
        dim_out = dim_out if dim_out else dim_in
        super().__init__(
            nn.Linear(dim_in, dim_hidden),
            DWConv(dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_out)
        )

class SpatialTrans(nn.Module):
    def __init__(self, dim, num_heads, s=4):
        super().__init__()

        assert dim%num_heads == 0

        self.scale = 1/math.sqrt(dim//num_heads)
        self.num_heads = num_heads
        self.ln = nn.LayerNorm(dim)
        self.r = nn.Conv3d(dim, dim, kernel_size=(1, s, s), stride=(1, s, s), bias=False)
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, 2*dim)
        self.w_out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, dim_hidden=4*dim)
        )

    def forward(self, x):

        # input is expected to be SAI B x C x N x H x W 
        b, _, _, h, _ = x.size()

        #perform self attention
        buffer = self.ln(rearrange(x, 'b c n h w -> (b n) (h w) c'))
        q = self.to_q(buffer)
        buffer = self.r(x)
        buffer = rearrange(buffer, 'b c n h w -> (b n) (h w) c')
        buffer = self.ln(buffer)
        k, v = self.to_kv(buffer).chunk(2, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads),
            [q, k, v]
        ) 
        A = F.softmax(self.scale * einsum(q, k, 'b h l1 d, b h l2 d -> b h l1 l2'), dim=-1)
        buffer = einsum(A, v, 'b h l1 l2, b h l2 d -> b h l1 d')

        #rearrange and fuse heads
        buffer = rearrange(buffer, 'b h l d -> b l (h d)')
        buffer = self.w_out(buffer) #how much impact does this mapping have?

        #residual connection
        x = rearrange(x, 'b c n h w -> (b n) (h w) c')
        x = buffer + x

        #mlp + residual connection
        x = x + self.mlp(x)

        #reshape
        x = rearrange(x, '(b n) (h w) c -> b c n h w', b=b, h=h)

        return x
    
class AngTrans(nn.Module):
    def __init__(self, dim, num_heads, m):
        super().__init__()

        self.m = m
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(dim//num_heads)

        self.ln = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3*dim)
        self.w_out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, dim_hidden=4*dim)
        )

    def forward(self, x, A=5):

        #input is expected to be SAI B x C x N x H x W 
        # reshape to batch of MacPis  

        x = rearrange(x, 'b c (u v) (h mh) (w mw) -> (b h w) (u mh v mw) c', u=A, mh=self.m, mw=self.m)

        buffer = self.ln(x)

        #self-attention
        q, k, v = map(
            lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads),
            self.to_qkv(x).chunk(3, dim=-1)
        )

        attn = F.softmax(self.scale * einsum(q, k, 'b h l1 d, b h l2 d -> b h l1 l2'), dim=-1)
        buffer = einsum(attn, v, 'b h l1 l2, b h l2 d -> b h l1 d')

        buffer = rearrange(buffer, 'b h l d -> b l (h d)')
        buffer = self.w_out(buffer)

        #residual connection
        x = x + buffer

        #mlp + res connect
        x = x + self.mlp(x)
        
        x = rearrange(x, '(b h w) (u mh v mw) c -> (b h mh w mw) (u v) c', u=A, mh=self.m, mw=self.m, h=32//self.m, w=32//self.m)

        return x
    
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Linear(3*dim, 3)

    def forward(self, xs):
        x = torch.cat(xs, dim=-1)
        a = F.softmax(self.fuse(x), dim=-1)
        x = rearrange(x, 'b l (h d) -> b l h d', h=3)
        x = einsum(a, x, 'b l h, b l h d -> b l d')
        return x
    
class AngCoder(nn.Module):
    def __init__(self, dim, num_heads, ms=[1, 2, 4]):
        super().__init__()
        self.encoders = nn.ModuleList([
            AngTrans(dim, num_heads, m) for m in ms
        ])
        self.fuse = AttentionFusion(dim)

    def forward(self, x):
        f = []
        for encoder in self.encoders:
            f.append(encoder(x))
        x = self.fuse(f) 
        # x is BHW x UV x C
        # reshape to B x C x N x H x W
        x = rearrange(x, '(b h w) n c -> b c n h w', h=32, w=32)
    
        return x


class ConvNet(nn.Sequential):
    def __init__(self, dim, dim_out=None):
        dim_out = dim_out if dim_out else dim
        super().__init__(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(dim, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(dim_out, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )
    

class FeatureAggregation(nn.Module):
    def __init__(self, dim, num_encoders=4):
        super().__init__()

        self.convs = nn.ModuleList([
            ConvNet(dim, dim_out=dim//2) for _ in range(num_encoders)
        ])
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, xs):
        x = 0
        for i, conv in enumerate(self.convs):
            x = x + xs[i] 
            xs[i] = self.lrelu(conv(x))

        return torch.cat(xs, dim=1)
    
    
class DET(nn.Module):
    def __init__(self, dim=64, num_heads=4, num_encoders=4, scale_factor=4):
        super().__init__()

        self.scale_factor = scale_factor

        ### shallow feature extraction
        self.in_conv = nn.Conv3d(1, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.conv = ConvNet(dim)

        ### deep feature extraction
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                SpatialTrans(dim, num_heads, s=2),
                AngCoder(dim, num_heads)
            ]) for _ in range(num_encoders)
        ])

        ### Hierarchical Feature Aggregation
        self.hfa = FeatureAggregation(dim, num_encoders=num_encoders)

        ### Upsampling
        self.up = nn.Sequential(
            nn.Conv2d(2*dim, scale_factor**2 * dim, kernel_size=1, bias=False),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        )

    def forward(self, x):
        
        #initial feature extraction
        buffer = self.in_conv(x.unsqueeze(1))
        buffer = buffer + self.conv(buffer)

        #dfe
        fs = []
        for spa, ang in self.encoders:
            buffer = ang(spa(buffer))
            fs.append(buffer)

        ###hfa
        buffer = self.hfa(fs)

        ### upsampling
        buffer = rearrange(buffer, 'b c n h w -> (b n) c h w')
        buffer = self.up(buffer)
        buffer = rearrange(buffer, '(b n) c h w -> (b c) n h w', n=25)

        return buffer + F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')