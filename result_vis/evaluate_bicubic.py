import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import LightFieldTestDataset
from utils.metrics import *

### Get the Data

#transformation

## mean and std of dataset
mean = (0.0403, 0.0527, 0.0594, 0.0673, 0.0781, 0.0874, 0.0926, 0.0936, 0.0957,
        0.0957, 0.0957, 0.0961, 0.1002, 0.1054, 0.1089, 0.1089, 0.1088, 0.1069,
        0.1054, 0.1056, 0.1097, 0.1130, 0.1148, 0.1158, 0.1180)
std = (0.0523, 0.0696, 0.0780, 0.0878, 0.1010, 0.1117, 0.1169, 0.1175, 0.1177,
        0.1156, 0.1132, 0.1125, 0.1165, 0.1242, 0.1319, 0.1352, 0.1343, 0.1317,
        0.1293, 0.1279, 0.1298, 0.1315, 0.1315, 0.1316, 0.1333)

## mean and std of dataset after normlaizing data with the above mean and std
## and then clamping the values to [-1, +1]
mean2 = (-0.1506, -0.1519, -0.1525, -0.1528, -0.1532, -0.1527, -0.1520, -0.1518,
        -0.1520, -0.1503, -0.1483, -0.1491, -0.1503, -0.1501, -0.1490, -0.1475,
        -0.1460, -0.1457, -0.1448, -0.1446, -0.1449, -0.1449, -0.1448, -0.1456,
        -0.1466)
std2 = (0.6029, 0.6081, 0.6144, 0.6195, 0.6274, 0.6305, 0.6338, 0.6341, 0.6425,
        0.6454, 0.6520, 0.6535, 0.6525, 0.6396, 0.6233, 0.6111, 0.6134, 0.6140,
        0.6173, 0.6241, 0.6358, 0.6480, 0.6595, 0.6665, 0.6722)

transform = transforms.Compose([
    transforms.Normalize(mean, std),
    transforms.Lambda(lambda x: torch.clamp(x, -1, 1)),
    transforms.Normalize(mean2, std2)
])

#Load Test Data
data_list = ['EPFL', 'HCI_new', 'HCI_old','INRIA_Lytro', 'Stanford_Gantry']
test_loaders = []
for data in data_list:
    dataset = LightFieldTestDataset([data], transform=transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    test_loaders.append(test_loader)

MODEL = 'bicubic'
exp_name = 'bicubic'

writer = SummaryWriter(f'runs/{MODEL}/training/{exp_name}')
metrics = ['SSIM', 'PSNR', 'SAM', 'SRE']

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

        _, _, h, w = LF.shape
        h -= h%4
        w -= w%4

        LF = LF[:, :, :h, :w]

        #sample donw
        LF_input = F.interpolate(LF, scale_factor=0.25, mode='bicubic')
        LF_target = LF

        #sample back up again
        LF_out = F.interpolate(LF_input, scale_factor=4, mode='bicubic')

        LF_out = LF_out.squeeze()
        LF_target = LF_target.squeeze()

        #PSNR and SSIM
        ssim_set[j], psnr_set[j] = compute_psnr_ssim(LF_out, LF_target)
        #SAM
        sam_set[j] = compute_sam(LF_out, LF_target)
        #SRE
        sre_set[j] = compute_sre(LF_out, LF_target)

    ssim[i] = ssim_set.mean()
    psnr[i] = psnr_set.mean()
    sam[i] = sam_set.mean()
    sre[i] = sre_set.mean()

for name, metric in zip(metrics, [ssim, psnr, sam, sre]):

    writer.add_scalars(
        name, dict(zip(data_list, metric)), global_step=1
    )

writer.add_scalars(
    'Avg', {'SSIM': ssim.mean(), 'PSNR': psnr.mean(), 'SAM': sam.mean(), 'SRE': sre.mean()}, global_step=1
)