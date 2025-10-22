import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import LightFieldDataset, LightFieldTestDataset

def normalize_hsi(x):

    x = np.clip(x, np.percentile(x, 0.5), np.percentile(x, 99.5))
    maxs = x.max(axis=(0, 1), keepdims=True)

    return x / maxs, maxs

def load_data(train_list, test_list, batch_size):

    transform = transforms.Compose([
        transforms.Lambda(lambda x: normalize_hsi(x)),
        transforms.ToTensor()
    ])

    #Load Train Data
    dataset = LightFieldDataset(train_list, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Load Test Data
    test_loaders = []
    for data in test_list:
        dataset = LightFieldTestDataset([data], transform=transform)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        test_loaders.append(test_loader)

    return train_loader, test_loaders