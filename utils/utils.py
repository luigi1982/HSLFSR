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