import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum

import math

from models.LFSR.epit import Residual, SpatialConv

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.a = math.log(10)

    def forward(self, wl):
        device = wl.device
        chn = torch.arange(0, self.dim//2, device=device)
        emb = wl[:, None] / torch.exp(self.a*chn/(self.dim//2))[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ExtraTokenAttention(nn.Module):

    def __init__(self, dim, emb_dim, num_heads=8):
        super().__init__()

        self.num_heads = num_heads

        self.in_ = nn.Linear(dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.to_qk = nn.Linear(emb_dim, 2*emb_dim)
        self.wave_length_emb = nn.Sequential(
            SinusoidalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )

        self.out = nn.Linear(emb_dim, dim)

        self.wls = nn.Parameter(
            torch.arange(0, 25),
            requires_grad=False
        )

        self.attn_mask = nn.Parameter(
            self.gen_mask(11),
            requires_grad=False
        )
        

    def gen_mask(self, k_w):

        '''
        mask so that tokens belonging to one wavelengths do not attend 
        to the wavelength specific token of other wavelengths
        - these are always the last token in the SAI -> v * 32
        '''

        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left

        mask = torch.zeros((5, 33, 5, 33))

        for v in range(5):
            for w in range(33):
                mask[v, w, :v, 32] = float("inf")
                mask[v, w, v+1:, 32] = float("inf")
                mask[v, w, :, max(0, w - k_w_left):min(w, w + k_w_right)] = float("inf")                

        mask = rearrange(mask, 'a1 b1 a2 b2 -> (a1 b1) (a2 b2)')

        return mask

    def forward(self, x):

        b, _, _, h, w = x.size()

        #expect x to be light field shaped B x C x UV x H x W
        wls = self.wave_length_emb(self.wls).unsqueeze(0).unsqueeze(-1)

        #expand wls is shape UV x C 
        #transpose and repeat H times to obtain UV x C x H 
        wls = wls.transpose(2, 1).repeat(b, 1, 1, h).unsqueeze(-1)
        x = torch.cat([x, wls], dim=-1)

        #reshape to epipolar lines
        #B x C x UV x H x W -> BUH x VW x C
        x = rearrange(x, 'b c (u v) h w ->  (b u h) (v w) c', u=5)
        x = self.in_(x)
        x = self.ln(x)
        q, k, v = map(
            lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads),
            [*(self.to_qk(x).chunk(2, dim=-1))] + [x]
        )

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self.attn_mask
        )

        out = rearrange(out, '(b u h) hs (v w) d -> b (hs d) (u v) h w', b=b, h=h, w=w+1)
        out = out[:, :, :, :, :w]
        out = rearrange(out, 'b d (u v) h w -> (b u h) (v w) d', u=5, v=5)

        out = self.mlp(out) + out
        out = self.out(out)

        out = rearrange(out, '(b u h) (v w) d -> b d (u v) h w', b=b, h=h, w=w)

        return out
    

class AltFilter(nn.Module):
    def __init__(self, channels, ang_res=5):
        super().__init__()
        self.ang_res = ang_res
        self.epi_trans = ExtraTokenAttention(channels, 2*channels)
        self.conv = SpatialConv(channels)

    def forward(self, x):
        
        '''
        Input is stacked SAI, B x UV x H x W x C
        For horizontal EPI feature extraction reshape to VW x BHU x C 
        (VW is first as batch_first=False is standard)
        For SpatialConv back to B x UV x H x W x C
        For vertcal EPI feature extraction to UH x BVW x C
        For SpatialConv back to B x UV x H x W x C

        Residual connections are added after both SpatialConvs
        '''

        shortcut = x
        [_, _, _, h, w] = x.size()

        #extract horizontal EPI features
        x = self.epi_trans(x)
        x = self.conv(x) + shortcut

        #extract vertical EPI features
        x = rearrange(x, 'b c (u v) h w -> b c (v u) w h', u=self.ang_res, v=self.ang_res)
        x = self.epi_trans(x)
        x = rearrange(x, 'b c (v u) w h -> b c (u v) h w', u=self.ang_res, v=self.ang_res, h=h, w=w)
        x = self.conv(x) + shortcut

        return x
    

class EPIT(nn.Module):
    def __init__(self, channels, ang_res=5, use_as_encoder=False):
        super().__init__()

        self.ang_res = ang_res
        self.use_as_encoder = use_as_encoder

        self.in_conv = nn.Sequential(
            nn.Conv3d(1, channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            Residual(SpatialConv(channels))
        )

        self.EPI_features = nn.Sequential(
            *[AltFilter(channels) for _ in range(5)]
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels*16, 1, bias=False),
            nn.PixelShuffle(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, 3, padding=1, bias=False)
        )

    def forward(self, x):
        [_, _, h, w] = x.size()
        x = self.in_conv(x.unsqueeze(1))
        x = self.EPI_features(x) + x

        # x is shape B x C x UV x H x W
        # reshape to B x C x UV x HW for upsampling
        x_up = rearrange(x, 'b c (u v) h w -> b c (u h) (v w)', u=self.ang_res, v=self.ang_res)
        x_up = self.upsampling(x_up)
        x_up = rearrange(x_up, 'b c (u h) (v w) -> b c (u v) h w', h=4*h, w=4*w, u=self.ang_res, v=self.ang_res)

        if self.use_as_encoder:
            return x_up, rearrange(x, 'b c (u v) h w -> b c (h u) (w v)', u=self.ang_res, v=self.ang_res)
        else:
            x_up = x_up.view((-1, 25, 128, 128))
            return x_up