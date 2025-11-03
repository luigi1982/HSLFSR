import torch
from torch.utils.data.dataset import Dataset
import os
import h5py
import numpy as np

### Dataset for SISR methods
### A datapoint consists of a single SAI

class LightFieldDataset(Dataset):
    def __init__(self, data_list, train=True, root_dir='data',
                 transform=True):
        super().__init__()

        self.transform = transform
        train = 'train' if train else 'test'
        self.data_dir = os.path.join(root_dir, train)
        self.file_list = []

        for data_name in data_list:
            path = os.path.join(root_dir, train, data_name)
            tmp_list = []
            for file in os.listdir(path):
                tmp_list.extend(25*[os.path.join(data_name, file)])

            self.file_list.extend(tmp_list)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.file_list[index])

        wl = index%25

        with h5py.File(path, 'r') as hf:
            LF = np.array(hf.get('LF'))[wl]

        if self.transform:
            LF = transform_train(LF, wl)

        return LF.unsqueeze(0)


class LightFieldTestDataset(Dataset):
    def __init__(self, data_list, transform=True, root_dir='../datasets'):
        super().__init__()

        self.transform = transform
        self.file_list = []

        for data_name in data_list:
            path = os.path.join(root_dir, data_name, 'test_hsi')
            tmp_list = os.listdir(path)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = os.path.join(root_dir, data_name, 'test_hsi', tmp_list[index])

            self.file_list.extend(tmp_list)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):

        with h5py.File(self.file_list[index], 'r') as hf:
            LF = np.array(hf.get('LF'))

        #LF = torch.from_numpy(LF).permute((2, 0, 1)).to(torch.float32)

        if self.transform is not None:
            LF = transform_test(LF)

        return LF

MAXS95 = [16.24786949157715, 24.152420043945312, 218.07720947265625, 254.61024475097656, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 250.62374877929688, 254.7626953125, 244.73486328125, 246.5926971435547, 189.19093322753906, 118.78125, 126.46803283691406, 79.95013427734375, 67.44583129882812]
MAXS95 = torch.tensor(MAXS95)

def transform_train(x, wl):
    x = torch.from_numpy(x).to(torch.float32)
    m = MAXS95[wl]
    x = torch.clamp(x, min=0, max=m)
    return x / m

def transform_test(x):

    x = torch.from_numpy(x).to(torch.float32)
    m = MAXS95.to(device=x.device, dtype=x.dtype).view(25, 1, 1)
    x = torch.clamp(x, min=torch.zeros_like(m), max=m)
    return x / m