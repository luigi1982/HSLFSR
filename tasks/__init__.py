from tqdm import tqdm
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torch
from datetime import datetime
import os
import h5py
from matplotlib import pyplot as plt

from utils.utils import *
from utils.metrics import *
from utils.data import load_data, denormalize_hsi

class Trainer():

    def __init__(
            self, exp_name, model_name, model, 
            train_data_list, test_data_list,
            epochs, device, batch_size,
            test_batch_size, 
            evaluation_step, save_lfs_step,
            optimizer=Adam, lr=2e-4, lr_decay_steps=15, gamma=0.5,
            mode='train',
            cross_val_run =None,
            start_epoch=0
        ):

        self.model = model
        self.model_name = model_name
        self.start_epoch=start_epoch
        self.epochs = epochs
        self.device = device
        self.test_batch_size = test_batch_size
        self.evaluation_step = evaluation_step
        self.save_lfs_step = save_lfs_step

        #load train data and load test data
        self.train_loader, self.test_loaders = self.load_datasets(
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
        ### and directories for saving models

        if mode == 'training':

            if start_epoch == 0:
                now = datetime.now()
                now = now.strftime('%m%d-%H%M')
                exp_name = exp_name+'-'+now 

            dir = f'runs/{model_name}/{mode}/{exp_name}' 
            save_model_path = ['models_', model_name, 'training', exp_name]

        elif mode == 'cross_val':     
            
            now = datetime.now()
            now = now.strftime('%m%d-%H%M')
            cross_val_run = cross_val_run+'-'+now 
            dir = f'runs/{model_name}/cross_val/{cross_val_run}/{exp_name}'
            save_model_path =  ['models_', model_name, mode, cross_val_run, exp_name]

        elif mode == 'evaluate':

            dir = f'runs/{model_name}/{mode}/{exp_name}'
            self.save_path = f'runs/{model_name}/{mode}/{exp_name}'

        self.writer = SummaryWriter(dir)
        self.metrics = ['SSIM', 'PSNR', 'SAM', 'SRE']
        self.best_psnr = 0
        self.data_list = test_data_list

        if mode in ['training', 'cross_val']:
            self.save_model_path = os.path.join(*save_model_path)

        ### saving LFs and models
        self.save_lfs_path = os.path.join(
            'results', model_name, mode, exp_name
        )

    def training(self):

        self.model.to(self.device)

        if 'cuda' in self.device:
            torch.cuda.set_device(self.device)

            # Enable TF32 for better performance when using bfloat16
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        for epoch in range(self.start_epoch, self.epochs):

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

    def evaluate(self, epoch, save_lfs, save_model=True, save_pV=False):

        #move model to device
        self.model.to(self.device)
        
        #loop over test data

        num_test = len(self.test_loaders)

        ssim = torch.zeros(num_test)
        psnr = torch.zeros(num_test)
        sam = torch.zeros(num_test)
        sre = torch.zeros(num_test)

        ssim_per_v = torch.zeros((num_test, 5, 5))
        psnr_per_v = torch.zeros((num_test, 5, 5))

        for i, test_loader in tqdm(enumerate(self.test_loaders), total=len(self.test_loaders)):

            ssim_set = torch.zeros(len(test_loader))
            psnr_set = torch.zeros(len(test_loader))
            sam_set = torch.zeros(len(test_loader))
            sre_set = torch.zeros(len(test_loader))

            ssim_per_v_set = torch.zeros((len(test_loader), 25))
            psnr_per_v_set = torch.zeros((len(test_loader), 25))
            
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
                ssim_set[j], psnr_set[j], ssim_per_v_set[j], psnr_per_v_set[j] = compute_psnr_ssim(LF_out, LF_target)
                #SAM
                sam_set[j] = compute_sam(LF_out, LF_target)
                #SRE
                sre_set[j] = compute_sre(LF_out, LF_target)

                ### save result if save_lfs True
                if save_lfs:
                    save_path = os.path.join(self.save_lfs_path, f'epoch_{epoch+1}', self.data_list[i])
                    os.makedirs(save_path, exist_ok=True)
                    with h5py.File(save_path+f'/scene_{j+1}.h5', 'w') as hf:
                        hf.create_dataset('SR', data=denormalize_hsi(LF_out).numpy())

            ssim[i] = ssim_set.mean()
            psnr[i] = psnr_set.mean()
            sam[i] = sam_set.mean()
            sre[i] = sre_set.mean()

            ssim_per_v[i] = torch.permute(ssim_per_v_set, (1, 0)).mean(dim=-1).view((5, 5))
            psnr_per_v[i] = torch.permute(psnr_per_v_set, (1, 0)).mean(dim=-1).view((5, 5))

        for name, metric in zip(self.metrics, [ssim, psnr, sam, sre]):

            self.writer.add_scalars(
                name, dict(zip(self.data_list, metric)), global_step=epoch+1
            )

        if save_pV:
            for name, metric in zip(['SSIM_pV', 'PSNR_pV'], [ssim_per_v, psnr_per_v]):
                path = os.path.join(self.save_path, 'per_view_statistics')
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f'{name}.h5')
                with h5py.File(path, 'w') as f:
                    for idx, data_name in enumerate(self.data_list):
                        f.create_dataset(name=data_name, data=metric[idx])

        print(
            f'SSIM: {ssim.mean():.3f}; PSNR: {psnr.mean():.3f}; SAM: {sam.mean():.3f}; SRE: {sre.mean():.3f}'
        )

        self.writer.add_scalars(
            'Avg', {'SSIM': ssim.mean(), 'PSNR': psnr.mean(), 'SAM': sam.mean(), 'SRE': sre.mean()}, global_step=epoch+1
        )

        ### save the model
        if psnr.mean() > self.best_psnr and save_model:
            print(
                f'Saving model - beat previously best PSNR by {psnr.mean() - self.best_psnr:.3f} dB'
            )
            self.best_psnr = psnr.mean()
            model_path = self.save_model_path
            os.makedirs(model_path, exist_ok=True)
            torch.save(self.model.state_dict(), model_path + f'/net_epoch_{epoch+1}.pth')

    def train_step(self, x):
        raise NotImplementedError 
    
    def evaluate_step(self, x):
        raise NotImplementedError 
    
    def load_datasets(self, train_data_list, test_data_list, batch_size, use_train_as_test=False):
        raise NotImplementedError
    
    

            