import torch
from torch import nn
from einops import rearrange, einsum   

class SwinBlock(nn.Module):
    def __init__(
            self, hidden_dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding
        ):
        super().__init__()

        #apply layer norm -> MSA -> res -> layer norm -> MLP -> res
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.msa = WindowAttention(hidden_dim, heads, head_dim, shifted, window_size, relative_pos_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )
        
    def forward(self, x):
        buffer = self.ln1(x) #.permute(0, 3, 1, 2)
        buffer = x + self.msa(buffer)
        buffer = self.ln2(buffer)
        return self.mlp(buffer) + x
    

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)

            self.upper_lower_mask = nn.Parameter(
                create_mask(window_size, displacement, upper_lower=True, left_right=False),
                requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(window_size, displacement, upper_lower=False, left_right=True),
                requires_grad=False
            )


        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(window_size**2, window_size**2))
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):

        if self.shifted:
            x = self.cyclic_shift(x)

        b, w_h, w_w, _, h = *x.shape, self.heads

        #get query, key and value matrix
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        #compute the number of windows
        nw_h = w_h // self.window_size
        nw_w = w_w // self.window_size

        #rearrange qkv
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size
            ),
            qkv
        )

        #dot product similarity
        dots = einsum(q, k, 'b h w i d, b h w j d -> b h w i j') * self.scale
        # add positional embeddings
        dots += self.pos_embedding
        #add the masking
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w-1::nw_w] += self.left_right_mask
        #compute attention scores
        attn = dots.softmax(dim=-1)
        #
        out = einsum(attn, v, 'b h w i j, b h w j d -> b h w i d')

        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                nw_h=nw_h, w_h=self.window_size, w_w=self.window_size
            )
        
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        return out
    
    
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))
    

def create_mask(window_size, displacement, upper_lower, left_right):

    mask = torch.zeros(window_size**2, window_size**2)

    if upper_lower:
        mask[-displacement*window_size:, :-displacement*window_size] = float('-inf') #down left section
        mask[:-displacement*window_size, -displacement*window_size:] = float('-inf') #up right section

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask