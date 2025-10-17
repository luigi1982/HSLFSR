import torch
from torch import nn
from torch.nn import functional as F
import math
from einops import rearrange
from tqdm import tqdm

def cosine_beta_schedule(T, s=8e-3):
    ts = torch.arange(0, T+1)
    ft = torch.cos((ts/T + s)/(1 + s)*torch.pi/2)**2
    alpha_bar = ft/ft[0]
    beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])

    return beta

class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, encoder_fn, timesteps=100):
        super().__init__()

        #set denoising function and low resolution image encoder
        self.denoise_fn = denoise_fn
        self.encoder_fn = encoder_fn

        self.timesteps = timesteps

        #define betas, alphas and alpha bar
        betas = cosine_beta_schedule(timesteps)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        #register as buffer so they are not optmized during training
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t):
        #get sample xt ~ q(xt|x0)
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bar[t][:, None, None, None, None]
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return xt, noise

    def forward(self, img_lr, img_hr, t=None, fix_encoder=True, use_enc_ups=True):

        #set x0 to be the high resolution image
        x0 = img_hr
        #get the batch size and the device
        b, device = x0.size(0), x0.device

        #if t is None sample timesteps
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=device).long()
        #else repeat timestep batch size times
        else:
            t = torch.LongTensor([t]).repeat(b).to(device)

        #compute conditioning using the encoder
        if fix_encoder:
            self.encoder_fn.eval()
            with torch.no_grad():
                enc_out, cond = self.encoder_fn(img_lr)

        else:
            enc_out, cond = self.encoder_fn(img_lr)

        img_lr_up = enc_out if use_enc_ups else F.interpolate(img_lr, scale_factor=4)
        
        #get the residual, 
        #subtract the preliminary upscaled lr image from the hr image, unsqueeze channel dimension
        x0 = self.img2res(x0.unsqueeze(1), img_lr_up)
        #sample q and get noise
        xt, noise = self.q_sample(x0, t)
        #predict noise
        #reshape xt to MacPi
        xt = rearrange(xt, 'b c (u v) h w -> b c (h u) (w v)', u=5, v=5)
        noise_pred = self.denoise_fn(xt, cond, t)

        return noise_pred, rearrange(noise, 'b c (u v) h w -> b c (h u) (w v)', u=5, v=5)
    
    @torch.no_grad()
    def p_sample(self, lr, scale=4, noise=None, use_enc_ups=True):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            device = next(self.denoise_fn.parameters()).device

            if noise is None:
                b, uv, h, w = lr.shape
                u, v = 2*[int(math.sqrt(uv))]
                x = torch.randn((b, 1, u*scale*h, v*scale*w)).to(device)
            else:
                x = noise

            #encode the input image
            enc_out, cond = self.encoder_fn(lr)
            img_lr_up = enc_out if use_enc_ups else F.interpolate(lr, scale_factor=scale)
            T = self.betas.size(0)
            n = x.size(0)
            for t in reversed(range(T)):
                ts = torch.tensor([t]).repeat(n).to(device)
                z = torch.randn_like(x) if t > 0 else 0
                eps = self.denoise_fn(x, cond, ts.to(torch.float32))
                x = (1/torch.sqrt(self.alphas[ts]))[:, None, None, None]*(x - ((1 - self.alphas[ts])/torch.sqrt(1 - self.alpha_bar[ts]))[:, None, None, None]*eps) + torch.sqrt(self.betas[ts])[:, None, None, None]*z

            x = x.view((-1, 1, u, 128, v, 128)).contiguous().permute((0, 1, 2, 4, 3, 5)).contiguous().view((-1, 1, u*v, 128, 128))

            return img_lr_up + x

    def img2res(self, x, img_lr_up, clip_input=True, res_rescale=2.0):
        x = (x - img_lr_up) * res_rescale
        if clip_input:
            x = x.clamp(-1, 1)
        return x




