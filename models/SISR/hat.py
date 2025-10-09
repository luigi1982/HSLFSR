### Hybrid Attention Transformer (HAT)

import torch
from torch import nn
from torch.nn import functional as F
from einops import einsum, rearrange
import math

from models.utils.swin import WindowAttention
from models.SISR.drcan import ChannelAttention

class HAT(nn.Module):
    def __init__(self, in_channels, channels, num_groups=6, num_blocks=6, scale=4):
        super().__init__()
        
        #initial convolution
        self.init_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        # RHAGs followed by a final convolutionaly layer
        self.body = nn.Sequential(
            *[RHAG(channels=channels, num_blocks=num_blocks) for _ in range(num_groups)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

        # upsampling
        self.up = nn.Sequential(
            nn.Conv2d(channels, scale**2 * channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = x + self.body(x)
        return self.up(x)


class RHAG(nn.Module):
    #Residual Hybrid Attention Group (RHAG)
    def __init__(self, channels, num_blocks=6):
        super().__init__()

        assert num_blocks%2==0, 'number of blocks must be divisable by 2'

        self.body = nn.Sequential(
            *[HAB(channels, i%2==1) for i in range(num_blocks)],
            OCAB(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.body(x) + x


class HAB(nn.Module):
    #Hybrid Attnetion Block (HAB)
    #RCAB and Shifted window attention are used in parallel
    def __init__(self, channels, shifted, alpha=0.01):
        super().__init__()

        window_size = 16
        heads = 4
        head_dim = channels // heads
        mlp_ratio = 2

        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)

        self.alpha = alpha

        self.cab = self.conv_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            ChannelAttention(channels)
        )

        self.wmsa = WindowAttention(
            channels, heads, head_dim, shifted, window_size, None
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_ratio*channels),
            nn.GELU(),
            nn.Linear(mlp_ratio*channels, channels)
        )

    def forward(self, x):

        # input x is B x C x H x W
        # permute to B x H x W x C
        x = x.permute((0, 2, 3, 1))
        buffer = self.ln1(x)
        x = x + self.wmsa(buffer) + self.alpha*self.cab(buffer.permute((0, 3, 1, 2))).permute((0, 2, 3, 1))
        buffer = self.ln2(x)
        x = self.mlp(buffer) + x

        return x.permute((0, 3, 1, 2))

class OCAB(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()

        mlp_ratio = 2

        self.ln1 = nn.LayerNorm(channels)
        self.oca = OCA(channels=channels, heads=heads)
        self.ln2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_ratio*channels),
            nn.GELU(),
            nn.Linear(mlp_ratio*channels, channels)
        )

    def forward(self, x):
        x = x.permute((0, 2, 3, 1))
        buffer = self.ln1(x).permute((0, 3, 1, 2))
        x = self.oca(buffer) + x
        buffer = self.ln2(x)
        x = self.mlp(buffer) + x
        return x.permute((0, 3, 1, 2))


class OCA(nn.Module):
    #Overlapping Cross-Attention
    def __init__(self, channels, heads=4, window_size=16, gamma=0.5):
        super().__init__()

        self.window_size=window_size
        self.gamma=gamma
        self.scale = math.log(channels)
        self.heads = heads

        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(channels, 2*channels)
        self.pos_embedding = nn.Parameter(torch.randn(window_size**2, int(window_size*(1+gamma))**2))

    def forward(self, x):

        #expects inputs BxCxHxW
        #returns outputs BxHxWxC

        _, c, h, w = x.shape

        #get patch size
        patch_size = int(self.window_size*(1 + self.gamma))
        delta = int(self.window_size * self.gamma)
        # split padding across both sides (allow odd splits)
        pad_h0 = delta // 2
        pad_h1 = delta - pad_h0
        pad_w0 = delta // 2
        pad_w1 = delta - pad_w0
        #get the number of patches
        hl, wl = h//self.window_size, w//self.window_size

        #pad and unfold for overlapping K and V
        x_pad = F.pad(x, (pad_w0, pad_w1, pad_h0, pad_h1))
        x_kv = F.unfold(
            x_pad, kernel_size=patch_size, stride=self.window_size
        )
        x_kv = x_kv.view((-1, c, patch_size, patch_size, hl*wl)).permute((0, 4, 2, 3, 1)).view((-1, patch_size**2, c))

        #unfold for Q
        x_q = F.unfold(
            x, kernel_size=self.window_size, stride=self.window_size
        )
        x_q = x_q.view((-1, c, self.window_size, self.window_size, hl*wl)).permute((0, 4, 2, 3, 1)).view((-1, self.window_size**2, c))

        #get query, key and value
        q = self.to_q(x_q)
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, 'b l (h c) -> b h l c', h=self.heads),
            [q, k, v]
        )

        #compute attention score
        attn = (einsum(q, k, 'b h l1 c, b h l2 c -> b h l1 l2') / self.scale).softmax(dim=-1)
        #add the positional embedding
        attn += self.pos_embedding
        #compute out
        out = einsum(attn, v, 'b h l1 l2, b h l2 c -> b h l1 c')
        #reshape
        out = rearrange(out, '(b l1 l2) h (p1 p2) c -> b (l1 p1) (l2 p2) (h c)', l1=hl, l2=wl, p1=self.window_size)

        return out

