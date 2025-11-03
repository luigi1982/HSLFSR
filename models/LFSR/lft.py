import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.a = math.log(1e4)

    def forward(self, x):
        device = x.device
        pos = torch.arange(x.size(1), device=device)
        chn = torch.arange(self.dim//2, device=device)
        chn = chn * self.a/(self.dim//2)
        emb = pos[:, None] / chn[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)[None, :, :]
    
class SinusoidalPosEmb2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.a = math.log(1e4)

    def forward(self, x):

        ### input is b l c
        ### assuming input is square -> l = h*w = h^2
        _, l, _ = x.size()
        device = x.device
        h = int(math.sqrt(l))
        pos = torch.arange(h, device=device)
        chn = torch.arange(self.dim//2, device=device)
        chn = chn * self.a/(self.dim//2)
        emb = pos[:, None] / chn[None, :]
        emb_cos = emb.cos()
        emb_sin = emb.sin()

        emb_sin = emb_sin[:, None, :] + emb_sin[None, :, :]
        emb_cos = emb_cos[:, None, :] + emb_cos[None, :, :]

        emb = torch.cat([emb_sin, emb_cos], dim=-1).view((1, -1, self.dim))

        return emb

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, pos_emb=SinusoidalPosEmb):
        super().__init__()

        self.pos_emb = pos_emb(dim)
        self.ln = nn.LayerNorm(dim)
        self.to_qk = nn.Linear(dim, 2*dim)
        self.mhsa = MHSA(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2*dim),
            nn.ReLU(),
            nn.Linear(2*dim, dim)
        )

    def forward(self, x, mask=None):
        t = self.ln(x + self.pos_emb(x))
        q, k = torch.chunk(self.to_qk(t), chunks=2, dim=-1)
        x = x + self.mhsa(q, k, t, mask=mask)
        x = x + self.mlp(x)
        return x

class MHSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.out = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        q, k, v = map(
            lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads),
            [q, k, v]
        )

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,     # shape broadcastable to [B, H, Lq, Lk]
            dropout_p=0.0,           # or self.attn_dropout during training
            is_causal=False          # set True for decoder-only causal attention
        )
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out(out)

        return out
    

class AngTrans(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.trans = TransformerBlock(dim, num_heads)

    def forward(self, x):
        ### expect the input as B x HW x UV x C
        ### return input as B x UV x HW x C <- can be processed with less effort by SpaTrans 
        b, n1, n2, c = x.size()
        x = rearrange(x, 'b n1 n2 c -> (b n1) n2 c')
        x = self.trans(x)
        x = rearrange(x, '(b n1) n2 c -> b n2 n1 c', n1=n1)
        return x


class SpaTrans(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=3, padding=1)
        self.trans = TransformerBlock(dim*2, num_heads*2, pos_emb=SinusoidalPosEmb2d)
        self.conv_out = nn.Conv2d(2*dim, dim, kernel_size=1)

        cached = self.gen_mask(32, 32, 5)
        self.register_buffer("attn_mask", cached, persistent=False)

    @staticmethod
    def gen_mask(h:int, w:int, k:int):

        '''
        Masking of the attention scores
        the sequences are EPIs so they are of shape U x H / V x W
        h -> angular dim, w -> spatial dim
        k_h controls the size of the neighbourhood in the angular domain
        k_w controls the size of the neighbourhood in the spatial domain
        '''    
        attn_mask = torch.zeros([h, w, h, w])
        k_left = k//2
        k_right = k - k_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i-k_left):min(h,i+k_right), max(0, j-k_left):min(h,j+k_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, x):
        ### expects the input as B x UV x HW x C
        b, n1, n2, c = x.size()
        h = int(math.sqrt(n2))
        x = rearrange(x, 'b n1 (h w) c -> (b n1) c h w ', h=h)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.trans(x, mask=self.attn_mask)
        x = rearrange(x, 'b (h w) c -> b c h w', h=32)
        x = self.conv_out(x)
        x = rearrange(x, '(b n1) c h w -> b (h w) n1 c', n1=n1)
        return x
    

class AltFilter(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.ang_trans = AngTrans(dim, num_heads)
        self.spa_trans = SpaTrans(dim, num_heads)

    def forward(self, x):
        x = self.ang_trans(x)
        x = self.spa_trans(x)
        return x
    

class LFT(nn.Module):

    def __init__(self, num_layers=4, dim=32, num_heads=4, scale=4):
        super().__init__()

        #initial conv
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        #body
        self.body = nn.Sequential(
            *[AltFilter(dim, num_heads) for _ in range(num_layers)]
        )

        #upsampling
        self.upsampling = nn.Sequential(
            nn.Conv2d(dim, dim*scale ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x, scale=4):
        #input B x UV x H x W

        #bc interpolate lr
        bc = F.interpolate(x, scale_factor=scale, mode='bicubic')
        
        #initial
        x = x.unsqueeze(1)
        buffer = self.conv_init0(x)
        x = x + self.conv_init(buffer)

        #body
        x = rearrange(x, 'b c n h w -> b (h w) n c')
        x = x + self.body(x)

        x = rearrange(x, 'b (h w) (u v) c -> b c (u h) (v w)', h=32, v=5)

        #upscale
        x = self.upsampling(x)
        x = rearrange(x, ' b c (u h) (v w) -> (b c) (u v) h w', h=scale*32, v=5)

        return x + bc
