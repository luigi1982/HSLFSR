import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

MAXS = [0.4479275345802307, 0.5814003348350525, 0.6595451831817627, 0.7216576337814331, 0.8344082832336426, 0.925993025302887, 0.9784007668495178, 1.0, 0.9981639385223389, 0.9780774712562561, 0.9672863483428955, 0.9659062623977661, 0.923708438873291, 0.9096832275390625, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99925297498703, 0.9989583492279053, 0.9732648730278015, 0.9420004487037659]
#MAXS95 = [0.2723314166069031, 0.3660186529159546, 0.4345605969429016, 0.50874263048172, 0.5811171531677246, 0.6348714232444763, 0.6647414565086365, 0.6708457469940186, 0.6647749543190002, 0.64559406042099, 0.6082891821861267, 0.58054518699646, 0.5877730250358582, 0.6491191387176514, 0.682621419429779, 0.6847525835037231, 0.6713098883628845, 0.6550964713096619, 0.6555171608924866, 0.6472086906433105, 0.654275119304657, 0.6720536351203918, 0.6838650107383728, 0.6861315369606018, 0.6902734637260437]
MAXS95 = [16.24786949157715, 24.152420043945312, 218.07720947265625, 254.61024475097656, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 250.62374877929688, 254.7626953125, 244.73486328125, 246.5926971435547, 189.19093322753906, 118.78125, 126.46803283691406, 79.95013427734375, 67.44583129882812]
MAXS = torch.tensor(MAXS)
MAXS95 = torch.tensor(MAXS95)

def normalize_hsi(x):

    #assert x.dim() == 3, "expected [C,H,W]"
    #c, h, w = x.shape
    m = MAXS95.to(device=x.device, dtype=x.dtype).view(25, 1, 1)
    x = torch.clamp(x, min=torch.zeros_like(m), max=m)
    return x / m

@torch.no_grad()
def denormalize_hsi(x_norm: torch.Tensor, m_vec: torch.Tensor = MAXS95) -> torch.Tensor:
    """
    x_norm: [C,H,W] in [0,1] after your normalize_hsi
    returns: [C,H,W] in original reflectance/radiance units (but still clipped at m)
    """
    assert x_norm.dim() == 3, "expected [C,H,W]"
    C, H, W = x_norm.shape
    m = m_vec.to(device=x_norm.device, dtype=x_norm.dtype).view(C,1,1)
    return x_norm * m

def load_data(train_list, test_list, batch_size, train_data, test_data, transform=True):

    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).to(torch.float32)),
        transforms.Lambda(normalize_hsi),
    ]) if transform else True

    #Load Train Data
    dataset = train_data(train_list, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Load Test Data
    test_loaders = []
    for data in test_list:
        dataset = test_data([data], transform=transform)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        test_loaders.append(test_loader)

    return train_loader, test_loaders


def load_data_sisr(train_list, test_list, batch_size, train_data, test_data, transform=True, sisr=False):

    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).to(torch.float32)),
        transforms.Lambda(normalize_hsi),
    ]) if transform else None

    #Load Train Data
    dataset = train_data(train_list, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Load Test Data
    test_loaders = []
    for data in test_list:
        dataset = test_data([data], transform=transform)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        test_loaders.append(test_loader)

    return train_loader, test_loaders
