from tasks import Trainer

import torch
from torch.optim import Adam
from torch.nn import functional as F

class LFSRTrainer(Trainer):

    def __init__(
            self, exp_name, model_name, model, 
            train_data_list, test_data_list,
            epochs, device, batch_size,
            test_batch_size, evaluation_step, save_lfs_step,
            criterion=torch.nn.L1Loss(),
            optimizer=Adam, lr=2e-4, lr_decay_steps=15, gamma=0.5
        ):

        super().__init__(
            exp_name, model_name, model, 
            train_data_list, test_data_list,
            epochs, device, batch_size,
            test_batch_size, 
            evaluation_step, save_lfs_step,
            optimizer=optimizer, lr=lr, lr_decay_steps=lr_decay_steps, gamma=gamma
        )

        self.criterion = criterion

    def train_step(self, hr):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            #sample down data
            lr = F.interpolate(hr, scale_factor=0.25, mode='bicubic')
            sr = self.model(lr)
            loss = self.criterion(sr, hr)

        return loss
    
    def evaluate_step(self, sub_LF_input):
        sub_LF_out = []
        for k in range(0, sub_LF_input.size(0), self.test_batch_size):
            tmp = sub_LF_input[k:min(k + self.test_batch_size, sub_LF_input.size(0)), :, :, :]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    self.model.eval()
                    torch.cuda.empty_cache()
                    out = self.model(tmp.to(self.device))
                    sub_LF_out.append(out)

        return sub_LF_out