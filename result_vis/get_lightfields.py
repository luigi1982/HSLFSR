import h5py
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F

def get_lfsr_paths(model, exp_name, degradation, 
                   datasets=['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors', 'multi_exposure_rec'],
                   num_scenes=1
    ):
    #for each test set get the first scene

    scene_paths = []

    path=os.path.join('results', model, 'evaluate', exp_name, degradation)
    
    for test in datasets:
        if not 'hold_out' in test:
            if num_scenes == 3:
                scenes = ['scene_6.h5', 'scene_7.h5', 'scene_5.h5']
                scenes=[os.path.join(path, test, scene) for scene in scenes]
                scene_paths = scene_paths + scenes
            elif num_scenes == 1:
                scenes = ['scene_1.h5']
                scenes=[os.path.join(path, test, scene) for scene in scenes]
                scene_paths = scene_paths + scenes
            else:
                scenes=os.listdir(os.path.join(path, test))[:num_scenes]
                scenes=[os.path.join(path, test, scene) for scene in scenes]
                scene_paths = scene_paths + scenes

    return scene_paths

def get_lfhr_paths(
        datasets=['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors', 'multi_exposure_rec'],
        num_scenes=1
):
    scene_paths = []
    path='../datasets'
    for test in datasets: #for test in os.listdir(path):
        test = os.path.join(path, test, 'test_hsi')
        if num_scenes == 3:
            scenes = ['scene_3.h5', 'scene_1.h5', 'scene_2.h5']
            scenes=[os.path.join(test, scene) for scene in scenes]
            scene_paths = scene_paths + scenes
        else:
            scenes=os.listdir(test)[:num_scenes]
            scenes=[os.path.join(test, scene) for scene in scenes]
            scene_paths = scene_paths + scenes

    return scene_paths

def load_lfs(paths, key):
    lfs = []
    for path in paths:
        with h5py.File(path, 'r') as hf:
            print(hf.keys())
            LF = np.array(hf.get(key))
        lfs.append(LF)

    return lfs

def plot_central_view(model, exp_name, srs, hrs):

    _, axs = plt.subplots(len(srs), 3, figsize=(6, 10), constrained_layout=True)
    for i, (sr, hr) in enumerate(zip(srs, hrs)):
        sr_cv = sr[12]
        hr_cv = hr[12]

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
    path=path+'/qualitative_results.pdf'
    plt.savefig(path, format='pdf', dpi=600)
    plt.close()

    return path

def create_qual_plot(model, exp_name):
        
    p2=get_lfsr_paths(model, exp_name)
    sigma = [4, 2, 0, 3, 1, 5]
    pr = [p2[i] for i in sigma]
    p2 = pr
    lfs_sr=load_lfs(p2, 'SR')
    p1=get_lfhr_paths()
    print(p1, p2)
    lfs_hr=load_lfs(p1, 'HR')
    path = plot_central_view(model, exp_name, lfs_sr, lfs_hr)

    return path
    
