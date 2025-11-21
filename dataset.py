import torch
from torch.utils.data.dataset import Dataset
import os
import h5py
import numpy as np

class LightFieldDataset(Dataset):
    def __init__(self, data_list, transform=None, train=True, root_dir='data'):
        super().__init__()
        self.transform = transform
        train = 'train' if train else 'test'
        self.data_dir = os.path.join(root_dir, train)
        self.file_list = []

        for data_name in data_list:
            path = os.path.join(root_dir, train, data_name)
            tmp_list = os.listdir(path)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = os.path.join(data_name, tmp_list[index])

            self.file_list.extend(tmp_list)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.file_list[index])

        with h5py.File(path, 'r') as hf:
            LF = np.array(hf.get('LF'))
            LF = LF.reshape((5, 5, 128, 128))
            LF[1::2] = LF[1::2, ::-1]
            LF = LF.reshape((25, 128, 128))

        if self.transform is not None:
            LF = self.transform(LF)

        return LF


class LightFieldTestDataset(Dataset):
    def __init__(self, data_list, transform=None, root_dir='../datasets'):
        super().__init__()
        self.transform = transform
        self.file_list = []

        for data_name in data_list:
            data = data_name.split('-')
            data_name = data[0]
            dataset = 'training_hsi' if len(data) > 1 else 'test_hsi'
            path = os.path.join(root_dir, data_name, dataset)
            tmp_list = os.listdir(path)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = os.path.join(root_dir, data_name, dataset, tmp_list[index])

            self.file_list.extend(tmp_list)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):

        with h5py.File(self.file_list[index], 'r') as hf:
            LF = np.array(hf.get('HR'))
            LF = LF.reshape((5, 5, 410, 410))
            LF[1::2] = LF[1::2, ::-1]
            LF = LF.reshape((25, 410, 410))

        #LF = np.transpose(LF, (2, 0, 1)) for synthetic datasets

        if self.transform is not None:
            LF = self.transform(LF)
            LF = LF.to(torch.float32)

        ### make sure that test-data s divisable into patches of size 32x32
        _, h, w = LF.size()
        h -= h%32
        w -= w%32
        LF = LF[:, :h, :w]

        return LF