import torch
from einops import rearrange
from torch.nn import functional as F


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]
    return Im_out

class LF_divide_integrate(object):
    def __init__(self, scale, patch_size, stride):
        self.scale = scale
        self.patch_size = patch_size
        self.stride = stride

        # bdr gives the size of the overlap between bordering patches
        self.bdr = (patch_size - stride) // 2
        self.pad = torch.nn.ReflectionPad2d(padding=(self.bdr, self.bdr + stride - 1, self.bdr, self.bdr + stride - 1))

    def LFdivide(self, LF):
        assert LF.size(0) == 1, 'The batch_size of LF for test requires to be one!'
        LF = LF.squeeze(0)
        [c, u, v, h, w] = LF.size()

        LF = rearrange(LF, 'c u v h w -> (c u v) 1 h w')

        #compute the number of patches in H and W direction
        self.numU = (h + self.bdr * 2 - 1) // self.stride
        self.numV = (w + self.bdr * 2 - 1) // self.stride

        # LF_pad = self.pad(LF)
        # add padding to the image
        LF_pad = ImageExtend(LF, [self.bdr, self.bdr + self.stride - 1, self.bdr, self.bdr + self.stride - 1])
        
        #overlapping patch extraction is performed
        LF_divided = F.unfold(LF_pad, kernel_size=self.patch_size, stride=self.stride)

        #batch the patches and reshape individual patches to SAIs, i.e. C x U x V x H x W
        LF_divided = rearrange(LF_divided, '(c u v) (h w) (numU numV) -> (numU numV) c u v h w', u=u, v=v,
                               h=self.patch_size, w=self.patch_size, numU=self.numU, numV=self.numV)
        return LF_divided

    def LFintegrate(self, LF_divided):

        #crop each patch in the last dimension
        LF_divided = LF_divided[:, :, :, :, self.bdr*self.scale:(self.bdr+self.stride)*self.scale,
                                self.bdr*self.scale:(self.bdr+self.stride)*self.scale]
        
        #reshape to a single LF
        LF = rearrange(LF_divided, '(numU numV) c u v h w -> c u v (numU h) (numV w)',
                       numU=self.numU, numV=self.numV)
        return LF