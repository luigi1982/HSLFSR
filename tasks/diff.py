import os
import sys
import torch
from torch.optim import Adam
from torch.nn import functional as F

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.data import load_data
from dataset import LightFieldDataset, LightFieldTestDataset
from tasks import Trainer
from diffusion.diffusion import GaussianDiffusion

class DIFFTrainer(Trainer):

    def __init__(
            self, exp_name, model_name, 
            denoise_fn, encoder_fn, 
            train_data_list, test_data_list,
            epochs, device, batch_size,
            test_batch_size, evaluation_step, save_lfs_step,
            criterion=torch.nn.L1Loss(),
            optimizer=Adam, lr=2e-4, lr_decay_steps=15, gamma=0.5,
            mode='training',
            cross_val_run=None,
            start_epoch=0,
            checkpoint=False
        ):

        model = GaussianDiffusion(denoise_fn, encoder_fn)

        if checkpoint:
            model.load_state_dict(torch.load(checkpoint, weights_only=True))

        for param in model.encoder_fn.parameters():
            param.requires_grad = False

        super().__init__(
            exp_name, model_name, model, 
            train_data_list, test_data_list,
            epochs, device, batch_size,
            test_batch_size, 
            evaluation_step, save_lfs_step,
            optimizer=optimizer, lr=lr, lr_decay_steps=lr_decay_steps, gamma=gamma,
            mode=mode,
            cross_val_run=cross_val_run,
            start_epoch=start_epoch
        )

        self.criterion = criterion

    def load_datasets(self, train_data_list, test_data_list, batch_size):
        return load_data(
            train_data_list, test_data_list, batch_size, train_data=LightFieldDataset, test_data=LightFieldTestDataset
        )

    def train_step(self, hr):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            #sample down data
            lr = F.interpolate(hr, scale_factor=0.25, mode='bicubic')

            #forward pass
            pred_noise, noise = self.model(lr, hr)

            #compute loss and update weights
            loss = self.criterion(pred_noise, noise)

        return loss
    
    def evaluate_step(self, sub_LF_input):
        sub_LF_out = []
        for k in range(0, sub_LF_input.size(0), self.test_batch_size):
            tmp = sub_LF_input[k:min(k + self.test_batch_size, sub_LF_input.size(0)), :, :, :]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    self.model.eval()
                    torch.cuda.empty_cache()
                    out = self.model.p_sample(tmp.to(self.device))
                    out = out.view((out.size(0), 25, 128, 128))
                    sub_LF_out.append(out)

        return sub_LF_out