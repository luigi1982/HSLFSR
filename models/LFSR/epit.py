import torch
from torch import nn
from einops import rearrange

class Residual(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x) + x

class BasicTrans(nn.Module):
    def __init__(self, channel, emb_dim, num_heads=8, dropout=.0):
        super().__init__()
        self.in_ = nn.Linear(channel, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.mhsa = nn.MultiheadAttention(emb_dim, num_heads, dropout, bias=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.out = nn.Linear(emb_dim, channel)

        cached = self.gen_mask(5, 32, 10, 11)
        self.register_buffer("attn_mask", cached, persistent=False)

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int):

        '''
        Masking of the attention scores
        the sequences are EPIs so they are of shape U x H / V x W
        h -> angular dim, w -> spatial dim
        k_h controls the size of the neighbourhood in the angular domain
        k_w controls the size of the neighbourhood in the spatial domain
        '''
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, x):

        '''
        Input is shape L x B x C
        Where L is sequence length, B batch size, C channel dim
        '''

        buffer = self.in_(x)
        buffer = self.norm(buffer)
        buffer = self.mhsa(
            query=buffer,
            key=buffer,
            value=buffer,
            attn_mask=self.attn_mask,
            need_weights=False
        )[0] + buffer
        buffer = self.mlp(buffer) + buffer
        buffer = self.out(buffer)

        return x + buffer
    
class SpatialConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1), bias=False)
        )

    def forward(self, x):
        return self.conv_net(x)
    
class AltFilter(nn.Module):
    def __init__(self, channels, ang_res=5):
        super().__init__()
        self.ang_res = ang_res
        self.epi_trans = BasicTrans(channels, 2*channels)
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
        x = rearrange(x, 'b c (u v) h w -> (v w) (b u h) c', u=self.ang_res, v=self.ang_res)
        x = self.epi_trans(x)
        x = rearrange(x, '(v w) (b u h) c -> b c (u v) h w', u=self.ang_res, v=self.ang_res, h=h, w=w)
        x = self.conv(x) + shortcut

        #extract vertical EPI features
        x = rearrange(x, 'b c (u v) h w -> (u h) (b v w) c', u=self.ang_res, v=self.ang_res)
        x = self.epi_trans(x)
        x = rearrange(x, '(u h) (b v w) c -> b c (u v) h w', u=self.ang_res, v=self.ang_res, h=h, w=w)
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
        
