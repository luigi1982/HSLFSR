import torch 
from torch import nn
from torch.nn import functional as F
from einops import rearrange, einsum

from models.LFSR.epit import SpatialConv, BasicTrans, Residual
         

class AltFilter(nn.Module):
    def __init__(self, channels, ang_res=5, shifted=False, version='v1'):
        super().__init__()
        self.shifted = shifted
        self.ang_res = ang_res
        self.epi_trans = EPISWinTBlock(channels, 2*channels)
        self.conv = SpatialConv(channels)
        self.version = version

    def v1(self, x):
        return self.epi_trans(x, shifted=self.shifted)

    def v2(self, x):
        x = self.epi_trans(x)
        x = self.epi_trans(x, shifted=True)
        return x
    
    def v(self, x):
        if self.version == 'v1':
            return self.v1(x)
        return self.v2(x)

    def forward(self, x):

        shortcut = x

        #extract horizontal EPI features
        x = self.v(x)
        x = self.conv(x) + shortcut

        #extract vertical EPI features
        x = rearrange(x, 'b c (u v) h w -> b c (v u) w h', u=self.ang_res, v=self.ang_res)
        x = self.v(x)
        x = rearrange(x, 'b c (v u) w h -> b c (u v) h w', u=self.ang_res, v=self.ang_res)
        x = self.conv(x) + shortcut

        return x


def cyclic_shift(x, shift):
    return torch.roll(x, -shift, -1)

def create_mask(p_h, p_w, shift):
    mask = torch.zeros((p_h, p_w, p_h, p_w))
    mask[:, :-shift, :, -shift:] = -float("inf")
    mask[:, -shift:, :, :-shift] = -float("inf")
    mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask

class EPISWinTBlock(nn.Module):

    def __init__(self, dim, emb_dim, num_heads=8, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.shift = patch_size // 2

        assert emb_dim%num_heads == 0
        self.num_heads = num_heads
        dim_head = dim // num_heads

        self.in_ = nn.Linear(dim, emb_dim)
        self.out = nn.Linear(emb_dim, dim)
        self.to_qkv = nn.Linear(emb_dim, 2*emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )

        self.pos_embeddings = nn.Parameter(
            torch.randn((5 * patch_size, 5 * patch_size))
        )

        self.attn_mask = nn.Parameter(
            create_mask(5, patch_size, self.shift),
            requires_grad=False
        )

    def forward(self, x, A=5, shifted=False):

        ### input Light Fields B x C x UV x x H x W
        ### reshape to Epipolar Lines (B x UH) x C x V x W

        b, _, _, h, _ = x.size()
        x = rearrange(x, 'b c (u v) h w -> (b u h) c v w', u=A)

        ### shift
        if shifted:
            x = cyclic_shift(x, self.shift)

        ### divide into local windows
        x = rearrange(x, 'b c v (p w_p) -> b p (v w_p) c', w_p=self.patch_size)

        ###pass each sequence through a transformer block
        x = self.in_(x)
        buffer = self.ln(x)
        qk = self.to_qkv(buffer)
        q, k, v = map(
            lambda t: rearrange(t, 'b p l (h d) -> b p h l d', h=self.num_heads),
            [*(qk.chunk(2, dim=-1)), x]
        )

        #mask if shifted
        if shifted:
            _, p, heads, l, _ = q.size()
            attn_mask = torch.zeros((p, heads, l, l), device=q.device)
            attn_mask[-1, :] = self.attn_mask
        buffer = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = None if not shifted else attn_mask
        )
        buffer = rearrange(buffer, 'b p h l d -> b p l (h d)')

        #residual connection 1
        x = x + buffer

        #mlp + res connect 2
        x = x + self.mlp(x)

        ### rearrange patches again and revert shifting
        x = self.out(x)
        x = rearrange(x, '(b u h) p (v w_p) c -> b c (u v) h (p w_p)', w_p=self.patch_size, b=b, h=h)
        x = cyclic_shift(x, -self.shift) if shifted else x
        
        return x
    

class EPIT(nn.Module):
    def __init__(self, channels, ang_res=5, version='v2', use_as_encoder=False):
        super().__init__()

        self.ang_res = ang_res
        self.use_as_encoder = use_as_encoder

        self.in_conv = nn.Sequential(
            nn.Conv3d(1, channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            Residual(SpatialConv(channels))
        )

        self.EPI_features = nn.Sequential(
            *[AltFilter(channels, shifted=(i%2==1), version=version) for i in range(3)]
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