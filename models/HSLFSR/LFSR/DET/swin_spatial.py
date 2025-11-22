import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum
import math

from models.LFSR.det import MLP, DET, AngCoder
from models.utils.swin import WindowAttention

class SpatialTrans(nn.Module):
    def __init__(self, dim, num_heads, shifted=False, window_size=8):
        super().__init__()

        assert dim%num_heads == 0

        self.swin = WindowAttention(
            dim, num_heads, dim//num_heads, shifted=shifted, window_size=window_size, relative_pos_embedding=None
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, dim_hidden=4*dim)
        )

    def forward(self, x):

        # input is expected to be SAI B x C x N x H x W 
        b, _, _, h, _ = x.size()

        #perform self attention
        x = rearrange(x, 'b c n h w -> (b n) h w c')
        x = self.swin(x)
        x = rearrange(x, 'b h w c -> b (h w) c')

        #mlp + residual connection
        x = x + self.mlp(x)

        #reshape
        x = rearrange(x, '(b n) (h w) c -> b c n h w', b=b, h=h)

        return x
    
class DET_SpatialSwin(DET):
    def __init__(self, dim=64, num_heads=4, num_encoders=4, scale_factor=4, window_size=8):
        super().__init__(dim, num_heads, num_encoders, scale_factor)
        
        ### deep feature extraction
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                SpatialTrans(dim, num_heads, shifted=(i+1)%2==0, window_size=window_size),
                AngCoder(dim, num_heads)
            ]) for i in range(num_encoders)
        ])