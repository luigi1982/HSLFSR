import torch
from torchmetrics.image import SpectralAngleMapper as SAM
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_sam(x, y):
    '''
    Input stacked SAIs x, y UV x H x W
    compute SAM of x, y
    '''
    sam = SAM()
    _, n, m = x.shape
    return sam(x.unsqueeze(0), y[:, :n, :m].unsqueeze(0))

def compute_sre(x, y):

    '''
    Input stacked SAIs x, y UV x H x W
    compute SRE of x, y
    '''

    c, n, m = x.size()
    x = x.contiguous().view((c, -1))
    y = y[:, :n, :m].contiguous().view((c, -1))

    sre = torch.norm(x - y, p=2, dim=-1).mean()

    return sre

def compute_psnr_ssim(x, target):
    c, _, _ = x.size()
    ssim_v = torch.zeros(c)
    psnr_v = torch.zeros(c)

    for i in range(c):
        n, m = x[i].numpy().shape
        ssim_v[i] = ssim(x[i].numpy(), target[i].numpy()[:n, :m], data_range=1)
        psnr_v[i] = psnr(x[i].numpy(), target[i].numpy()[:n, :m], data_range=1)

        if np.isinf(psnr_v[i]):
            psnr_v[i] = 60

    return ssim_v.mean(), psnr_v.mean(), ssim_v, psnr_v
