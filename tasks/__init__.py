from tqdm import tqdm
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torch
from datetime import datetime
import os
import h5py

from dataset import LightFieldDataset, LightFieldTestDataset

from utils.utils import *
from utils.metrics import *
from utils.data import load_data

class Trainer():

    def __init__(
            self, exp_name, model_name, model, 
            train_data_list, test_data_list,
            epochs, device, batch_size,
            test_batch_size, 
            evaluation_step, save_lfs_step,
            optimizer=Adam, lr=2e-4, lr_decay_steps=15, gamma=0.5
        ):

        self.model = model
        self.epochs = epochs
        self.device = device
        self.test_batch_size = test_batch_size
        self.evaluation_step = evaluation_step
        self.save_lfs_step = save_lfs_step

        #load train data and oad test data
        self.train_loader, self.test_loaders = load_data(
            train_data_list, test_data_list, batch_size
        ) 

        #set the optimizer
        self.opt = optimizer(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, 
            step_size=lr_decay_steps, 
            gamma=gamma
        )

        ### Set up Tensorboard
        name = exp_name
        now = datetime.now()
        now = now.strftime('%m%d-%H%M')
        exp_name = name+'-'+now
        writer = SummaryWriter(f'runs/{model_name}/training/{exp_name}')
        metrics = ['SSIM', 'PSNR', 'SAM', 'SRE']

    def training(self):

        if 'cuda' in self.device:
            torch.cuda.set_device(self.device)

            # Enable TF32 for better performance when using bfloat16
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        for epoch in range(self.epochs):

            running_loss = 0.0

            for i, hr in tqdm(
                    enumerate(self.train_loader), total=len(self.train_loader)
                ):

                # set model to training
                self.model.train()

                #zero the gradients
                self.opt.zero_grad()

                #pass data to device
                hr = hr.to(self.device)

                #perform training step
                loss = self.train_step(hr)

                #perform backprop
                loss.backward()
                self.opt.step()

                running_loss += loss.item()

            self.scheduler.step()

            print(
                f'[{epoch+1}/{self.epochs}] Loss: {running_loss/len(self.train_loader):.3f}'
            )

            #track metrics
            self.writer.add_scalar(
                'Train Loss', running_loss/len(self.train_loader), global_step=epoch+1
            )

            if (epoch+1)%self.evaluation_step == 0:
                save_lfs = (epoch+1)%self.save_lfs_step == 0
                self.evaluate(epoch, save_lfs)

    def evaluate(self, epoch, save_lfs):
        
        #loop over test data

        num_test = len(self.test_loaders)

        ssim = torch.zeros(num_test)
        psnr = torch.zeros(num_test)
        sam = torch.zeros(num_test)
        sre = torch.zeros(num_test)

        for i, test_loader in tqdm(enumerate(self.test_loaders), total=len(self.test_loaders)):

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
                sub_LF_out = self.evaluate_step(sub_LF_input)

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
                if save_lfs:
                    save_path = os.path.join('results', self.model_name, self.exp_name, f'epoch_{epoch+1}', self.data_list[i])
                    os.makedirs(save_path, exist_ok=True)
                    with h5py.File(save_path+f'/scene_{j+1}.h5', 'w') as hf:
                        hf.create_dataset('SR', data=LF_out.numpy())

    def train_step(self, x):
        raise NotImplementedError 
    
    def evaluate_step(self, x):
        raise NotImplementedError 




            