import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from jsonargparse import ArgumentParser, ActionConfigFile
import h5py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import LightFieldTestDataset
from utils.utils import *
from utils.metrics import *
from utils.data import load_data
from config import TrainConfig, make_serializable

from models.LFSR.distg_unet import DISTG_UNET
from models.LFSR.epit import EPIT
from diffusion import GaussianDiffusion

model='distg_unet'
exp='run1-1014-1858'

parser = ArgumentParser(
    description="Training script with config files",
    default_config_files=[f"results/{model}/{exp}/config.yaml"]
)
parser.add_argument("--config", action=ActionConfigFile)
parser.add_class_arguments(TrainConfig, nested_key="train")
cfg = parser.parse_args()

EPOCHS=cfg.train.epochs
BS=cfg.train.batch_size
DEVICE=cfg.train.device
MODEL=cfg.train.model.model
ENC_PATH='models_/epit/dim32-1014-1552/net_epoch_50.pth'

print(f'Running on device: {DEVICE}')

### Get the Data
data_list = ['EPFL', 'HCI_new', 'HCI_old','INRIA_Lytro', 'Stanford_Gantry']
_, test_loaders = load_data(data_list, data_list, 1)

#get models
denoise_fn = DISTG_UNET(32, 32, 32).to(DEVICE)
encoder_fn = EPIT(32, use_as_encoder=True).to(DEVICE)

#initialize diffusion model
f = GaussianDiffusion(denoise_fn, encoder_fn, timesteps=100).to(DEVICE)

#tensorboard
writer = SummaryWriter(f'runs/{model}/training/{exp}')
metrics = ['SSIM', 'PSNR', 'SAM', 'SRE']

num_test = len(test_loaders)

ssim = torch.zeros(num_test)
psnr = torch.zeros(num_test)
sam = torch.zeros(num_test)
sre = torch.zeros(num_test)

for epoch in range(5, EPOCHS+1, 5):

    #load model
    path = f'models_/{model}/{exp}/net_epoch_{epoch}.pth'
    f.load_state_dict(torch.load(path, weights_only=True))

    for i, test_loader in tqdm(enumerate(test_loaders), total=len(test_loaders)):

        ssim_set = torch.zeros(len(test_loader))
        psnr_set = torch.zeros(len(test_loader))
        sam_set = torch.zeros(len(test_loader))
        sre_set = torch.zeros(len(test_loader))
        
        for j, LF in enumerate(test_loader):

            LF_input = F.interpolate(LF, scale_factor=0.25, mode='bicubic')
            LF_target = LF

            #Crop LFs into Patches
            LF_divide_integrate_func = LF_divide_integrate(4, 32, 16)
            sub_LF_input = LF_divide_integrate_func.LFdivide(LF_input)

            batch_size = 64

            #SR the Patches
            sub_LF_out = []
            for k in range(0, sub_LF_input.size(0), batch_size):
                tmp = sub_LF_input[k:min(k + batch_size, sub_LF_input.size(0)), :, :, :]
                with torch.no_grad():
                    f.eval()
                    torch.cuda.empty_cache()
                    out = f.p_sample(tmp.to(DEVICE))
                    out = out.view((out.size(0), 25, 128, 128))
                    sub_LF_out.append(out)

            #fuse patches back together
            sub_LF_out = torch.cat(sub_LF_out, dim=0)
            LF_out = LF_divide_integrate_func.LFintegrate(sub_LF_out).unsqueeze(0)
            LF_out = LF_out[:, :, 0:LF_target.size(-2), 0:LF_target.size(-1)].cpu().detach()

            ### compute metrics

            LF_out = LF_out.squeeze()
            LF_target = LF_target.squeeze()

            #PSNR and SSIM
            ssim_set[j], psnr_set[j] = compute_psnr_ssim(LF_out, LF_target)
            #SAM
            sam_set[j] = compute_sam(LF_out, LF_target)
            #SRE
            sre_set[j] = compute_sre(LF_out, LF_target)

            save_path = os.path.join('results', model, exp, f'epoch_{epoch}', data_list[i])
            os.makedirs(save_path, exist_ok=True)
            with h5py.File(save_path+f'/scene_{j+1}.h5', 'w') as hf:
                hf.create_dataset('SR', data=LF_out.numpy())


        ssim[i] = ssim_set.mean()
        psnr[i] = psnr_set.mean()
        sam[i] = sam_set.mean()
        sre[i] = sre_set.mean()

    for name, metric in zip(metrics, [ssim, psnr, sam, sre]):

        writer.add_scalars(
            name, dict(zip(data_list, metric)), global_step=epoch
        )

    print(
        f'[{epoch}/{EPOCHS}] | SSIM: {ssim.mean():.3f}; PSNR: {psnr.mean():.3f}; SAM: {sam.mean():.3f}; SRE: {sre.mean():.3f}'
    )

    writer.add_scalars(
        'Avg', {'SSIM': ssim.mean(), 'PSNR': psnr.mean(), 'SAM': sam.mean(), 'SRE': sre.mean()}, global_step=epoch
    )