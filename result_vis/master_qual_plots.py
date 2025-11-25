import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch.nn import functional as F


def create_single_qual_plot(gt_path, sr_paths, models, number):

    with h5py.File(gt_path, 'r') as f:
        gt = np.array(f['HR'])

    srs = []

    for path in sr_paths:
        with h5py.File(path, 'r') as f:
            sr = np.array(f['SR'])
            srs.append(sr)

    gt = gt.reshape((5, 5, 410, 410))
    gt[1::2] = gt[1::2, ::-1]
    gt = gt.reshape((25, 410, 410))

    main_img = gt[12]

    # Layout -------------------------------------------------------------
    fig = plt.figure(figsize=(15, 4))
    gs = fig.add_gridspec(3, len(srs) + 3, width_ratios=[2] + [1, 1] + [1]*len(srs))

    # --- LEFT IMAGE -----------------------------------------------------
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(main_img, cmap='gray')
    ax_main.axis("off")

    # Green crop rectangle
    crop_x, crop_y, crop_w, crop_h = 100, 100, 120, 120  # modify
    ax_main.add_patch(Rectangle(
        (crop_x, crop_y), crop_w, crop_h,
        edgecolor='lime', linewidth=3, fill=False
    ))

    # Green crop rectangle
    epi_x, epi_y, epi_w, epi_h = 30, 120, 100, 2  # modify
    ax_main.add_patch(Rectangle(
        (epi_x, epi_y), epi_w, epi_h,
        edgecolor='blue', linewidth=1, fill=False
    ))

    # Bicubic Downsampling
    x = torch.from_numpy(gt)
    x = F.interpolate(x.unsqueeze(0), scale_factor=0.25, mode='bicubic')
    bc_up = F.interpolate(x, scale_factor=4, mode='bicubic')
    bc_up = bc_up.squeeze().numpy()

    # --- CROP PATCHES ---------------------------------------------------
    for i, (sr, name) in enumerate(zip([gt, bc_up] + srs,  ['Ground Truth', 'Bicubic'] + models)):
        crop = sr[12][crop_x: crop_x+crop_w, crop_y: crop_y+ crop_h]
        ax = fig.add_subplot(gs[:2, i + 1])
        ax.imshow(crop, cmap="gray")
        ax.axis("off")

        # add method name
        ax.set_title(name)

        # Green border around crop
        for spine in ax.spines.values():
            spine.set_edgecolor("lime")
            spine.set_linewidth(2)

        epi = sr[5:10, epi_y, epi_x:epi_x+epi_w].reshape((5, 100))
        ax = fig.add_subplot(gs[2, i + 1])
        ax.imshow(epi, aspect='auto', cmap='gray')
        ax.axis("off")

    plt.tight_layout()
    path = f'result_vis/master_plots/qualitative_plot-{number}.pdf'
    plt.savefig(path, dpi=1200)

    return path