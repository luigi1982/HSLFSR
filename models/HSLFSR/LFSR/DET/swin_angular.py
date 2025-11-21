import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum
import math

from models.LFSR.det import ConvNet, SpatialTrans, AngTrans, FeatureAggregation, MLP, AttentionFusion

def create_mask(m, displacement, upper_lower, left_right, A=5):

    l = (m*A)**2
    mask = torch.zeros(l, l)

    if upper_lower:
        mask = rearrange(mask, '(mu1 u1 mv1 v1) (mu2 u2 mv2 v2) -> mu1 u1 mv1 v1 mu2 u2 mv2 v2', u1=5, mu1=m, v1=5, mv1=m, u2=5, mu2=m, v2=5, mv2=m)
        mask[-displacement:, :, :, :, :-displacement, :, :, :] = float("inf")
        mask[:-displacement, :, :, :, -displacement:, :, :, :] = float("inf")
        mask = rearrange(mask, 'mu1 u1 mv1 v1 mu2 u2 mv2 v2 -> (mu1 u1 mv1 v1) (mu2 u2 mv2 v2)')

    if left_right:
        mask = rearrange(mask, '(mu1 u1 mv1 v1) (mu2 u2 mv2 v2) -> mu1 u1 mv1 v1 mu2 u2 mv2 v2', u1=5, mu1=m, v1=5, mv1=m, u2=5, mu2=m, v2=5, mv2=m)
        mask[:, :, -displacement:, :, :, :, :-displacement, :] = float("inf")
        mask[:, :, :-displacement, :, :, :, -displacement:, :] = float("inf")
        mask = rearrange(mask, 'mu1 u1 mv1 v1 mu2 u2 mv2 v2 -> (mu1 u1 mv1 v1) (mu2 u2 mv2 v2)')

    return mask


class CyclicShift(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return torch.roll(x, shifts=(self.shift, self.shift), dims=(3, 4))

class AngTrans(nn.Module):
    def __init__(self, dim, num_heads, m, shifted=False):
        super().__init__()

        self.m = m
        self.shifted = shifted
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(dim//num_heads)

        self.ln = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3*dim)
        self.w_out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, dim_hidden=4*dim)
        )

        if shifted:
            self.cyclic_shift = CyclicShift(-m//2)
            self.cyclic_back_shift = CyclicShift(m//2)

            self.upper_lower_mask = nn.Parameter(
                create_mask(m, m//2, upper_lower=True, left_right=False),
                requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(m, m//2, upper_lower=False, left_right=True),
                requires_grad=False
            )

    def forward(self, x, A=5):

        #input is expected to be SAI B x C x N x H x W 
        # reshape to batch of MacPis  

        if self.shifted:
            x = self.cyclic_shift(x)

        x = rearrange(x, 'b c (u v) (h mh) (w mw) -> (b h w) (u mh v mw) c', u=A, mh=self.m, mw=self.m)

        buffer = self.ln(x)

        #self-attention
        q, k, v = map(
            lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads),
            self.to_qkv(x).chunk(3, dim=-1)
        )

        attn = self.scale * einsum(q, k, 'b h l1 d, b h l2 d -> b h l1 l2')
        if self.shifted:
            attn = rearrange(attn, '(b h w) hds l1 l2 -> b h w hds l1 l2 ', h=32//self.m, w=32//self.m)
            attn[:, -1, :, :] =  attn[:, -1, :, :] + self.upper_lower_mask
            attn[:, :, -1, :] = attn[:, :, -1, :] + self.left_right_mask
            attn = rearrange(attn, 'b h w hds l1 l2 -> (b h w) hds l1 l2')
        attn = F.softmax(attn, dim=-1)
        buffer = einsum(attn, v, 'b h l1 l2, b h l2 d -> b h l1 d')

        buffer = rearrange(buffer, 'b h l d -> b l (h d)')
        buffer = self.w_out(buffer)

        #residual connection
        x = x + buffer

        #mlp + res connect
        x = x + self.mlp(x)

        if self.shifted:
            x = rearrange(x, '(b h w) (u mh v mw) c -> b c (u v) (h mh) (w mw)', u=A, v=A, mh=self.m, mw=self.m, h=32//self.m, w=32//self.m)
            x = self.cyclic_back_shift(x)
            x = rearrange(x, 'b c n (h mh) (w mw) -> (b h mh w mw) n c', mh=self.m, mw=self.m)
        else:
            x = rearrange(x, '(b h w) (u mh v mw) c -> (b h mh w mw) (u v) c', u=A, mh=self.m, mw=self.m, h=32//self.m, w=32//self.m)

        return x
    
    
class AngCoder(nn.Module):
    def __init__(self, dim, num_heads, ms=[1, 2, 4], shifted=False):
        super().__init__()
        self.encoders = nn.ModuleList([
            AngTrans(dim, num_heads, m, shifted=(m > 1 and shifted)) for m in ms
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
                AngCoder(dim, num_heads, shifted=(i+1)%2==0)
            ]) for i in range(num_encoders)
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