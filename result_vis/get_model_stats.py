import torch
from torchprofile import profile_macs
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from diffusion.diffusion import GaussianDiffusion
from models import MODEL_REGISTRY

def get_flops(model):
    x = torch.randn(1, 25, 32, 32)

    with torch.no_grad():
        macs = profile_macs(model, x)
    
    return 2*macs

def get_num_params(model):
    return sum(p.numel() for p in model.parameters())

def get_model_stats(model_name, dim=64):
    if model_name == 'distg_unet':
        denoise_fn = MODEL_REGISTRY['distg_unet'](32, 32, 32)
        encoder_fn = MODEL_REGISTRY['epit'](32, use_as_encoder=True)
        model = GaussianDiffusion(denoise_fn, encoder_fn)
        num_params = get_num_params(model)
        num_flops = 0
    else:
        model = MODEL_REGISTRY[model_name](dim)
        num_params = get_num_params(model)
        num_flops = get_flops(model)

    return num_params, num_flops

if __name__ == '__main__':
    model_name = 'epit'
    print(get_model_stats(model_name))