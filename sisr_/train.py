import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os
import h5py
import numpy as np
import yaml
from jsonargparse import namespace_to_dict

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import LightFieldDataset, LightFieldTestDataset
from utils.utils import *
from utils.metrics import *

from models import MODEL_REGISTRY

### parsing of config
from jsonargparse import ArgumentParser, ActionConfigFile
from config import TrainConfig, make_serializable

parser = ArgumentParser(description="Training script with config files")
parser.add_argument("--config", action=ActionConfigFile)
parser.add_class_arguments(TrainConfig, nested_key="train")
cfg = parser.parse_args()

EPOCHS=cfg.train.epochs
BS=cfg.train.batch_size
DEVICE=cfg.train.device
MODEL=cfg.train.model.model
EVAL_BS=cfg.train.evaluation.batch_size


### Get Training Data

#transformation for train data
mean = (0.0403, 0.0527, 0.0594, 0.0673, 0.0781, 0.0874, 0.0926, 0.0936, 0.0957,
        0.0957, 0.0957, 0.0961, 0.1002, 0.1054, 0.1089, 0.1089, 0.1088, 0.1069,
        0.1054, 0.1056, 0.1097, 0.1130, 0.1148, 0.1158, 0.1180)
std = (0.0523, 0.0696, 0.0780, 0.0878, 0.1010, 0.1117, 0.1169, 0.1175, 0.1177,
        0.1156, 0.1132, 0.1125, 0.1165, 0.1242, 0.1319, 0.1352, 0.1343, 0.1317,
        0.1293, 0.1279, 0.1298, 0.1315, 0.1315, 0.1316, 0.1333)
mean2 = (-0.1506, -0.1519, -0.1525, -0.1528, -0.1532, -0.1527, -0.1520, -0.1518,
        -0.1520, -0.1503, -0.1483, -0.1491, -0.1503, -0.1501, -0.1490, -0.1475,
        -0.1460, -0.1457, -0.1448, -0.1446, -0.1449, -0.1449, -0.1448, -0.1456,
        -0.1466)
std2 = (0.6029, 0.6081, 0.6144, 0.6195, 0.6274, 0.6305, 0.6338, 0.6341, 0.6425,
        0.6454, 0.6520, 0.6535, 0.6525, 0.6396, 0.6233, 0.6111, 0.6134, 0.6140,
        0.6173, 0.6241, 0.6358, 0.6480, 0.6595, 0.6665, 0.6722)
transform = [mean, std, mean2, std2]

#load the train data
data_list = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']
dataset = LightFieldDataset(data_list=data_list, transform=transform)
train_loader = DataLoader(dataset, batch_size=BS, shuffle=True)


### Get Test Data

#transformation for test data
transform = transforms.Compose([
    transforms.Normalize(mean, std),
    transforms.Lambda(lambda x: torch.clamp(x, -1, 1)),
    transforms.Normalize(mean2, std2)
])

#transform for Fraunhofer data
fh_mean = [77.50871276855469, 94.1110610961914, 103.54187774658203, 113.3948974609375, 123.1492919921875, 122.4135971069336, 118.8858413696289, 117.47124481201172, 114.16205596923828, 103.87840270996094, 53.53489685058594, 35.38602066040039, 32.56996154785156, 41.59749984741211, 46.3492546081543, 13.224126815795898, 12.063109397888184, 12.69680404663086, 13.029623985290527, 11.775398254394531, 7.188027381896973, 6.724524974822998, 6.2527031898498535, 6.291234493255615, 6.388221263885498]
fh_std = [65.06198120117188, 69.64328002929688, 73.69134521484375, 77.8589096069336, 80.6109619140625, 69.97572326660156, 70.15408325195312, 70.6668472290039, 71.26215362548828, 71.0075454711914, 57.78847885131836, 37.27056884765625, 35.62845993041992, 48.57426834106445, 50.59513473510742, 6.511099338531494, 6.36549711227417, 7.854125022888184, 7.298714637756348, 5.340404033660889, 0.760065495967865, 0.5303599238395691, 0.4376065135002136, 0.41120827198028564, 0.448697954416275]
fh_transform = transforms.Compose([
    transforms.Normalize(fh_mean, fh_std)
])

#load the test data
data_list = ['EPFL', 'HCI_new', 'HCI_old','INRIA_Lytro', 'Stanford_Gantry', 'Fraunhofer']
test_loaders = []
for data in data_list[:-1]:
    dataset = LightFieldTestDataset([data], transform=transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    test_loaders.append(test_loader)

dataset = LightFieldTestDataset([data_list[-1]], transform=transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
test_loaders.append(test_loader)


### Load Model
model_class = MODEL_REGISTRY[cfg.train.model.model]
net = model_class().to(DEVICE)


### Optimizer
opt = torch.optim.Adam(
    net.parameters(), 
    lr=cfg.train.optim.lr, 
    betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.StepLR(
    opt, 
    step_size=cfg.train.optim.lr_decay_steps, 
    gamma=cfg.train.optim.gamma
)


### Define the loss
criterion = torch.nn.L1Loss()


### Set up Tensorboard
name = cfg.train.name
now = datetime.now()
now = now.strftime('%m%d-%H%M')
exp_name = name+'-'+now
writer = SummaryWriter(f'runs/{MODEL}/training/{exp_name}')
metrics = ['SSIM', 'PSNR', 'SAM', 'SRE']


### save the config
save_path = os.path.join('results', MODEL, exp_name)
os.makedirs(save_path, exist_ok=True)
cfg_dict = make_serializable(cfg.as_dict())
with open(save_path + '/config.yaml', 'w') as file:
    yaml.safe_dump(cfg_dict, file, sort_keys=False)


### Train Loop

if 'cuda' in DEVICE:
    torch.cuda.set_device(DEVICE)
    # Enable TF32 for better performance when using bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

for epoch in range(EPOCHS):
    net.train()
    running_loss = 0.0
    for i, hr in tqdm(enumerate(train_loader), total=len(train_loader)):

        if isinstance(hr, list):
            hr[0] = hr[0].to(DEVICE)
        else:
            hr = hr.to(DEVICE)

        #pass through model
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):

            #sample down data
            lr = F.interpolate(hr, scale_factor=0.25, mode='bicubic')
            sr = net(lr)
            loss = criterion(sr, hr)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running_loss += loss.item()

    scheduler.step()

    print(f'[{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.3f}')

    #track metrics
    writer.add_scalar(
        'Train Loss', running_loss/len(train_loader), global_step=epoch+1
    )

    #loop over test data

    num_test = len(test_loaders)

    ssim = torch.zeros(num_test)
    psnr = torch.zeros(num_test)
    sam = torch.zeros(num_test)
    sre = torch.zeros(num_test)

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

            #SR the Patches
            sub_LF_out = []
            for k in range(0, sub_LF_input.size(0), 1):
                tmp = sub_LF_input[k:min(k + 1, sub_LF_input.size(0)), :, :, :]
                tmp = tmp.view((25, 1, 32, 32))
                outs = []
                for l in range(0, tmp.size(0), EVAL_BS):
                    sais = tmp[l:l+EVAL_BS]
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        with torch.no_grad():
                            net.eval()
                            torch.cuda.empty_cache()
                            out = net(sais.to(DEVICE))
                            outs.append(out)
                
                sub_LF_out.append(torch.cat(outs, dim=0).view((1, 25, 128, 128)))

            #fuse patches back together
            sub_LF_out = torch.cat(sub_LF_out, dim=0)
            LF_out = LF_divide_integrate_func.LFintegrate(sub_LF_out).unsqueeze(0)
            LF_out = LF_out[:, :, 0:LF_target.size(-2), 0:LF_target.size(-1)].cpu().to(torch.float32).detach()

            ### compute metrics

            LF_out = LF_out.squeeze()
            LF_target = LF_target.squeeze()

            #PSNR and SSIM
            ssim_set[j], psnr_set[j] = compute_psnr_ssim(LF_out, LF_target)
            #SAM
            sam_set[j] = compute_sam(LF_out, LF_target)
            #SRE
            sre_set[j] = compute_sre(LF_out, LF_target)

            ### save result every 5 epochs
            if (epoch+1)%5 == 0:
                save_path = os.path.join('results', MODEL, exp_name, f'epoch_{epoch+1}', data_list[i])
                os.makedirs(save_path, exist_ok=True)
                with h5py.File(save_path+f'/scene_{j+1}.h5', 'w') as hf:
                    hf.create_dataset('SR', data=LF_out.numpy())


        ssim[i] = ssim_set.mean()
        psnr[i] = psnr_set.mean()
        sam[i] = sam_set.mean()
        sre[i] = sre_set.mean()

    for name, metric in zip(metrics, [ssim, psnr, sam, sre]):

        writer.add_scalars(
            name, dict(zip(data_list, metric)), global_step=epoch+1
        )

    print(
        f'SSIM: {ssim.mean():.3f}; PSNR: {psnr.mean():.3f}; SAM: {sam.mean():.3f}; SRE: {sre.mean():.3f}'
    )

    writer.add_scalars(
        'Avg', {'SSIM': ssim.mean(), 'PSNR': psnr.mean(), 'SAM': sam.mean(), 'SRE': sre.mean()}, global_step=epoch+1
    )

    ### save the model
    model_path = os.path.join('models_', MODEL, exp_name)
    os.makedirs(model_path, exist_ok=True)
    torch.save(net.state_dict(), model_path + f'/net_epoch_{epoch+1}.pth')