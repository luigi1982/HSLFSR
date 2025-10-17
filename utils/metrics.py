import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_sam(x, y):
    '''
    Input stacked SAIs x, y UV x H x W
    compute SAM of x, y
    '''
    c, _, _ = x.size()
    x = x.permute((1, 2, 0)).contiguous().view((-1, c))
    y = y.permute((1, 2, 0)).contiguous().view((-1, c))
    epsilon = 1e-8  # Small value to prevent division by zero
    denominator = torch.norm(x, p=2, dim=-1) * torch.norm(y, p=2, dim=-1) + epsilon
    sam = torch.arccos((x*y).sum(-1)/denominator).mean()

    return sam

def compute_sre(x, y):

    '''
    Input stacked SAIs x, y UV x H x W
    compute SRE of x, y
    '''

    c, _, _ = x.size()
    x = x.contiguous().view((c, -1))
    y = y.contiguous().view((c, -1))

    sre = torch.norm(x - y, p=2, dim=-1).mean()

    return sre

def compute_psnr_ssim(x, target):
    c, _, _ = x.size()
    ssim_v = torch.zeros(c)
    psnr_v = torch.zeros(c)
    for i in range(c):
        ssim_v[i] = ssim(x[i].numpy(), target[i].numpy(), data_range=1)
        psnr_v[i] = psnr(x[i].numpy(), target[i].numpy(), data_range=1)

    return ssim_v.mean(), psnr_v.mean()
