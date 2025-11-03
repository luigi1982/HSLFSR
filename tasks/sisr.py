import torch
from torch.optim import Adam
from torch.nn import functional as F

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tasks import Trainer
from utils.data import load_data
from sisr_.dataset import LightFieldDataset, LightFieldTestDataset

class SISRTrainer(Trainer):

    def __init__(
            self, exp_name, model_name, model, 
            train_data_list, test_data_list,
            epochs, device, batch_size,
            test_batch_size, evaluation_step, save_lfs_step,
            criterion=torch.nn.L1Loss(),
            optimizer=Adam, lr=2e-4, lr_decay_steps=15, gamma=0.5,
            cross_val_run=None
        ):

        super().__init__(
            exp_name, model_name, model, 
            train_data_list, test_data_list,
            epochs, device, batch_size,
            test_batch_size, 
            evaluation_step, save_lfs_step,
            optimizer=optimizer, lr=lr, lr_decay_steps=lr_decay_steps, gamma=gamma,
            cross_val_run=cross_val_run
        )

        self.criterion = criterion

    def load_datasets(self, train_data_list, test_data_list, batch_size):
        return load_data(
            train_data_list, test_data_list, batch_size, train_data=LightFieldDataset, test_data=LightFieldTestDataset, transform=False
        )

    def train_step(self, hr):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):

            #sample down data
            lr = F.interpolate(hr, scale_factor=0.25, mode='bicubic')
            sr = self.model(lr)
            loss = self.criterion(sr, hr)

        return loss
    
    def evaluate_step(self, sub_LF_input):
        sub_LF_out = []
        for k in range(0, sub_LF_input.size(0), 1):
            tmp = sub_LF_input[k:min(k + 1, sub_LF_input.size(0)), :, :, :]
            tmp = tmp.view((25, 1, 32, 32))
            outs = []
            for wl in range(0, tmp.size(0), self.test_batch_size):
                sais = tmp[wl:wl+self.test_batch_size]
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        self.model.eval()
                        torch.cuda.empty_cache()
                        out = self.model(sais.to(self.device), wl)
                        outs.append(out)
            
            sub_LF_out.append(torch.cat(outs, dim=0).view((1, 25, 128, 128)))

        return sub_LF_out