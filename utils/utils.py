import torch
from torch.nn import functional as F
from einops import rearrange

class LF_divide_integrate(object):
    def __init__(self, scale, patch_size, stride):
        self.scale = scale
        self.patch_size = patch_size
        self.stride = stride
        self.bdr = (patch_size - stride) // 2
        self.pad = torch.nn.ReflectionPad2d(padding=(self.bdr, self.bdr + stride - 1, self.bdr, self.bdr + stride - 1))

    def LFdivide(self, LF):
        assert LF.size(0) == 1, 'The batch_size of LF for test requires to be one!'
        LF = LF.squeeze(0)
        [n, h, w] = LF.size()

        LF = LF.unsqueeze(1)
        self.numU = (h + self.bdr * 2 - 1) // self.stride
        self.numV = (w + self.bdr * 2 - 1) // self.stride

        LF_pad = self.pad(LF)
        LF_divided = F.unfold(LF_pad, kernel_size=self.patch_size, stride=self.stride)
        LF_divided = rearrange(LF_divided, 'n (h w) (numU numV) -> (numU numV) n h w',
                               h=self.patch_size, w=self.patch_size, numU=self.numU, numV=self.numV)
        return LF_divided

    def LFintegrate(self, LF_divided):
        LF_divided = LF_divided[:, :, self.bdr*self.scale:(self.bdr+self.stride)*self.scale,
                                self.bdr*self.scale:(self.bdr+self.stride)*self.scale]
        LF = rearrange(LF_divided, '(numU numV) n h w -> n (numU h) (numV w)',
                       numU=self.numU, numV=self.numV)
        return LF
    

def gaussian_kernel(ksize=21, sigma=2.0, device="cpu"):
    """
    Create 2D Gaussian kernel.

    Args:
        ksize (int): kernel size (odd).
        sigma (float): standard deviation of Gaussian.
    """
    assert ksize % 2 == 1, "ksize should be odd"

    ax = torch.arange(ksize, device=device) - (ksize - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def classical_degradation(x, scale_factor=0.25, noise_std=0.01):

    # 1. blur
    # 2. sample down
    # 3. add noise

    B, C, H, W = x.shape

    # blur

    blur_kernel = gaussian_kernel(device=x.device)

    kH, kW = blur_kernel.shape[-2:]
    pad_y = (kH - 1) // 2
    pad_x = (kW - 1) // 2

    x = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    # Apply depthwise convolution (same kernel per channel)
    # blur_kernel: (1,1,kH,kW) -> repeat for each channel
    kernel = blur_kernel.repeat(C, 1, 1, 1)   # (C,1,kH,kW)
    x = F.conv2d(x, kernel, groups=C)


    # sample down

    x = F.interpolate(x, scale_factor=scale_factor, mode='bicubic')


    # add noise
    noise = torch.randn_like(x) * noise_std
    x = x + noise

    x = x.clamp(0, 1)

    return x