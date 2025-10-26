import torch
from torch import nn
import math
from torch.nn import functional as F
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    ## p_{k,2i} = sin(k/(10000^(2i/d)))
    ## p_{k,2i+1} = cos(k/(10000^(2i/d)))
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        print(x[:, None].shape, emb[None, :].shape)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(
            dim, dim, 3, 2, 1
        )

    def forward(self, x):
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            dim, dim, 4, 2, 1
        )

    def forward(self, x):
        return self.conv(x)


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
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)