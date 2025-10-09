import torch
from torch import nn
from tqdm import tqdm

from models.LFSR.distg_unet import NET
from models.LFSR.epit import EPIT
from diffusion.diffusion import GaussianDiffusion

ENC_PATH = ''
BS = 16
EPOCHS = 80
DEVICE = 'cuda:0'
LR = 2e-4

#load data
dataset = None
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BS, shuffle=True)

#get models
denoise_fn = NET(32, 32, 32).to(DEVICE)
encoder_fn = EPIT(32, use_as_encoder=True).to(DEVICE)

#load encoder model
#encoder_fn.load_state_dict(torch.load(ENC_PATH, weights_only=True))

#initialize diffusion model
f = GaussianDiffusion(denoise_fn, encoder_fn, timesteps=100)

#set the optimizer
opt = torch.optim.Adam(f.parameters(), lr=LR)

#set loss function
criterion = torch.nn.L1Loss()

#tensorboard

#train loop
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        #zero the gradients
        opt.zero_grad()

        #move batch to device
        batch = batch.to(DEVICE)

        #degrade batch
        #to be implemented

        #forward pass
        pred_noise, noise = f(batch)

        #compute loss and update weights
        loss = criterion(pred_noise, noise)
        loss.backward()
        opt.step()

        running_loss += loss.item()

    print(f'[{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader)}')
        