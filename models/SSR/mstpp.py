import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum
import math

class MST(nn.Module):
    def __init__(self, channels=25, num_stages=3):
        super().__init__()

        #initial convolution
        self.init_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        #SST
        self.body = nn.Sequential(
            *[SST((1, 1, 1), channels) for _ in range(num_stages)],
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

        #upsampling
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels*16, 1, bias=False),
            nn.PixelShuffle(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, 3, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = x + self.body(x)
        return self.up(x)
    

class SST(nn.Module):
    def __init__(self, num_blocks, dim):
        super().__init__()

        dim_stage = dim

        #input projection
        self.init_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)

        #contracting path
        self.encoder = nn.ModuleList([])

        for i in range(2):
            self.encoder.append(
                nn.ModuleList([
                    MSAB(num_blocks=num_blocks[i], dim=dim_stage, num_heads=dim_stage//dim),
                    nn.Conv2d(dim_stage, 2*dim_stage, kernel_size=4, stride=2, padding=1, bias=False)
                ])
            )
            dim_stage *= 2

        #bottleneck
        self.mid = MSAB(num_blocks=num_blocks[2], dim=dim_stage, num_heads=dim_stage//dim)

        #expanding path
        self.decoder = nn.ModuleList([])

        for i in range(2):
            self.decoder.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(dim_stage, dim_stage//2, kernel_size=2, stride=2, bias=False),
                    nn.Conv2d(dim_stage, dim_stage//2, kernel_size=1, bias=False),
                    MSAB(num_blocks=num_blocks[-(i+1)], dim=dim_stage//2, num_heads=dim_stage//(2*dim))
                ])
            )
            dim_stage //= 2

        #output projection
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)      

    def forward(self, x):

        x = self.init_conv(x)
        h = []
        
        #contracting path
        for attn, down in self.encoder:
            x = attn(x)
            h.append(x)
            x = down(x)

        #bottleneck
        x = self.mid(x)

        #expanding path
        for up, conv, attn in self.decoder:
            x = conv(torch.concat([up(x), h.pop()], dim=1))
            x = attn(x)

        return self.out_conv(x)
    
class MSAB(nn.Sequential):
    def __init__(self, num_blocks, dim, num_heads):
        super().__init__(
            *[SAB(dim, num_heads) for _ in range(num_blocks)]
        )
    

class SAB(nn.Module):
    def __init__(self, dim, num_heads, mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.smsa = SMSA(dim, num_heads)
        self.mlp = FFN(dim, mult)

    def forward(self, x):
        _, c, h, w = x.shape
        x = x.permute((0, 2, 3, 1)).view((-1, h*w, c))
        buffer = self.ln1(x)
        x = self.smsa(buffer) + x
        buffer = self.ln2(x)
        buffer = buffer.view((-1, h, w, c)).permute((0, 3, 1, 2))
        buffer = self.mlp(buffer)

        return buffer + x.view((-1, h, w, c)).permute((0, 3, 1, 2))

    
class FFN(nn.Sequential):
    def __init__(self, dim, mult):
        super().__init__(
            nn.Conv2d(dim, mult*dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mult*dim, mult*dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mult*dim, dim, kernel_size=1, bias=False)
        )

class SMSA(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()

        assert c % num_heads == 0, 'Channels must be divisable by the number of heads'

        #attention
        self.num_heads = num_heads
        self.to_qkv = nn.Linear(c, 3*c)
        self.sigma = nn.Parameter(torch.ones(num_heads))
        self.merge_heads = nn.Linear(c, c)

        #positional embedding
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)
        )

    def forward(self, x):

        _, n, _ = x.shape
        height, width = 2*[int(math.sqrt(n))]

        #compute attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h c) -> b h n c', h=self.num_heads), qkv
        )
        attn = self.sigma[None, :, None, None] * einsum(k, q, 'b h n c1, b h n c2 -> b h c1 c2')
        attn = F.softmax(attn, dim=-1)
        head = einsum(v, attn, 'b h n c2, b h c1 c2 -> b h n c1')
        head = rearrange(head, 'b h n c -> b n (h c)')
        out = self.merge_heads(head)

        #compmute positional embeddings
        v = rearrange(v, 'b h (hg wd) c -> b (h c) hg wd', hg=height, wd=width)
        pos = self.ffn(v)
        pos = rearrange(pos, 'b c h w -> b (h w) c')

        return out + pos
