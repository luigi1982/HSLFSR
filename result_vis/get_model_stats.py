import torch
from fvcore.nn import FlopCountAnalysis
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from diffusion.diffusion import GaussianDiffusion
from models import MODEL_REGISTRY

def get_flops(model, x=None):
        
    if x is None:
        x = torch.randn(1, 25, 32, 32)

    device = 'cuda:2'
    model.to(device)
    x = x.to(device)

    flops = FlopCountAnalysis(model, x)
    
    return flops.total()

def get_num_params(model):
    return sum(p.numel() for p in model.parameters())

def get_forward_pass_time(model, x=None):

    if x is None:
        x = torch.randn(1, 25, 32, 32)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    device = 'cuda:2'
    model.to(device)
    x = x.to(device)

    avg_ms = 0

    for i in range(10):

        starter.record()
        with torch.no_grad():
            out = model(x)
        ender.record()

        torch.cuda.synchronize()              # Wait for GPU
        elapsed_ms = starter.elapsed_time(ender)

        avg_ms = elapsed_ms/(i+1) + i*avg_ms/(i+1)

    return avg_ms
    

def get_model_stats(model_name, dim=64):
    if model_name == 'distg_unet':
        denoise_fn = MODEL_REGISTRY['distg_unet'](32, 32, 32)
        encoder_fn = MODEL_REGISTRY['epit'](32, use_as_encoder=True)
        model = GaussianDiffusion(denoise_fn, encoder_fn)
        num_params = get_num_params(model)

        forward_pass_time_encoder = get_forward_pass_time(encoder_fn)
        num_flops_encoder = get_flops(encoder_fn)

        get_stats_model = MODEL_REGISTRY['distg_unet_get_stats'](32, 32, 32)
        forward_pass_time_unet = 100*get_forward_pass_time(get_stats_model, x=torch.randn((1, 32, 25, 32, 32)))
        num_flops_unet = 100*get_flops(get_stats_model, x=torch.randn((1, 32, 25, 32, 32)))

        num_flops = num_flops_encoder + num_flops_unet
        forward_pass_time = forward_pass_time_encoder + forward_pass_time_unet

    elif model_name in ['drcan', 'swinir', 'hat']:
        model = MODEL_REGISTRY[model_name](dim)
        num_params = get_num_params(model)
        x = torch.randn((1, 1, 32, 32))
        num_flops = 25*get_flops(model, x=x)
        forward_pass_time = 25*get_forward_pass_time(model, x=x)

    else:
        model = MODEL_REGISTRY[model_name](dim)
        num_params = get_num_params(model)
        num_flops = get_flops(model)
        forward_pass_time = get_forward_pass_time(model)

    return num_params, num_flops, forward_pass_time

if __name__ == '__main__':
    model_name = 'epit'
    print(get_model_stats(model_name))