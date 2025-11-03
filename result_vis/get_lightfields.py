import h5py
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F

def get_lfsr_paths(model, exp_name):
    #for each test set get the first scene

    scene_paths = []

    path=os.path.join('results', model, exp_name)
    epochs=os.listdir(path)
    nums=[]
    for epoch in epochs:
        try:
            _, num = epoch.split('_')
            nums.append(int(num))
        except:
            pass
    epoch=max(nums)
    path=os.path.join(path, 'epoch_'+str(epoch))
    
    for test in os.listdir(path):
        if not 'hold_out' in test:
            scene = os.path.join(path, test, 'scene_1.h5')
            scene_paths.append(scene)

    return scene_paths

def get_lfhr_paths():
    scene_paths = []
    path='../datasets'
    for test in os.listdir(path):
        test = os.path.join(path, test, 'test_hsi')
        scene=os.listdir(test)[0]
        scene=os.path.join(test, scene)
        scene_paths.append(scene)

    return scene_paths

def load_lfs(paths, key):
    lfs = []
    for path in paths:
        with h5py.File(path, 'r') as hf:
            LF = np.array(hf.get(key))
        lfs.append(LF)

    return lfs

def plot_central_view(model, exp_name, srs, hrs):

    _, axs = plt.subplots(len(srs), 3, figsize=(6, 10), constrained_layout=True)
    for i, (sr, hr) in enumerate(zip(srs, hrs)):
        sr_cv = sr[12]
        hr_cv = np.transpose(hr, (2, 0, 1))[12]

        #apply first normalization and clamping to HR image
        #revert second normalization on SR image

        mean1 = 0.0961
        std1 = 0.1125
        mean2 = -0.1483
        std2 = 0.6535

        hr_cv = (hr_cv - mean1) / std1
        sr_cv = std2*sr_cv + mean2

        '''_, axs1 = plt.subplots(1, 2)
        hr_u8 = (255*(hr_cv + 1)/2).astype(np.uint8)
        sr_u8 = (255*(sr_cv + 1)/2).astype(np.uint8)
        x = np.arange(256)
        hist_hr, _ = np.histogram(hr_u8, bins=256, range=(0, 256))
        hist_sr, _ = np.histogram(sr_u8, bins=256, range=(0, 256))
        axs1[0].bar(x, hist_hr)
        axs1[1].bar(x, hist_sr)

        plt.show()
        plt.savefig(f'bins_{i}.png')'''

        print('Min - Max', sr_cv.min(), sr_cv.max())

        bc = torch.from_numpy(hr_cv).unsqueeze(0).unsqueeze(0)
        bc = F.interpolate(bc, scale_factor=0.25, mode='bicubic')
        bc = F.interpolate(bc, scale_factor=4, mode='bicubic').squeeze().numpy()

        axs[i][0].imshow(bc, cmap="gray")
        axs[i][1].imshow(sr_cv, cmap="gray")
        axs[i][2].imshow(hr_cv, cmap="gray")

        # Remove ticks/axes
        for j in range(3):
            axs[i][j].axis("off")

    plt.show()
    path=f'result_vis/imgs/{model}/{exp_name}'
    os.makedirs(path, exist_ok=True)
    path=path+'/qualitative_results.png'
    plt.savefig(path)
    plt.close()

    return path

def create_qual_plot(model, exp_name):
        
    p2=get_lfsr_paths(model, exp_name)
    lfs_sr=load_lfs(p2, 'SR')
    p1=get_lfhr_paths()
    p1 = p1 if len(lfs_sr) == 6 else p1[1:]
    lfs_hr=load_lfs(p1, 'LF')
    path = plot_central_view(model, exp_name, lfs_sr, lfs_hr)

    return path
    
