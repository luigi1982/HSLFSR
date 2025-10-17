import torch
from torch.utils.data.dataset import Dataset
import os
import h5py
import numpy as np

### Dataset for SISR methods
### A datapoint consists of a single SAI

class LightFieldDataset(Dataset):
    def __init__(self, data_list, train=True, root_dir='../transformer_for_HSLFSR/data',
                 transform=None):
        super().__init__()

        if transform is not None:
            self.transform = True
            self.mean1, self.std1, self.mean2, self.std2 = transform
        else:
            self.transform = False

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

        LF = torch.from_numpy(LF)
        if self.transform:
            LF = (LF - self.mean1[wl]) / self.std1[wl]
            LF = torch.clamp(LF, min=-1, max=1)
            LF = (LF - self.mean2[wl]) / self.std2[wl]

        return LF.unsqueeze(0)


class LightFieldTestDataset(Dataset):
    def __init__(self, data_list, transform=None, root_dir='../datasets'):
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

        LF = torch.from_numpy(LF).permute((2, 0, 1)).to(torch.float32)

        if self.transform is not None:
            LF = self.transform(LF)

        return LF