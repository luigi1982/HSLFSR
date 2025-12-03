import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch.nn import functional as F


def create_single_qual_plot(gt_path, sr_paths, models, number, degradation, custom_box_coords=None):

    with h5py.File(gt_path, 'r') as f:
        try:
            gt = np.array(f['HR'])
        except:
            gt = np.array(f['LF'])

    srs = []

    for path in sr_paths:
        with h5py.File(path, 'r') as f:
            sr = np.array(f['SR'])
            srs.append(sr)

    print(gt.shape)

    gt = gt.reshape((5, 5, 410, 410))
    gt[1::2] = gt[1::2, ::-1]
    gt = gt.reshape((25, 410, 410))

    main_img = gt[12]

    # Layout -------------------------------------------------------------
    fig = plt.figure(figsize=(15, 4))
    gs = fig.add_gridspec(3, len(srs) + 2 + 1, width_ratios=[2] + [1] + [1]*len(srs) + [1])

    # --- LEFT IMAGE -----------------------------------------------------
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(main_img, cmap='gray')
    ax_main.axis("off")

    if custom_box_coords is None:
        crop_x, crop_y, crop_w, crop_h = 100, 100, 120, 120
    else:
        crop_x, crop_y, crop_w, crop_h = custom_box_coords

    # Green crop rectangle
    ax_main.add_patch(Rectangle(
        (crop_x, crop_y), crop_w, crop_h,
        edgecolor='lime', linewidth=3, fill=False
    ))

    # --- CROP PATCHES ---------------------------------------------------
    for i, (sr, name) in enumerate(zip([gt] + srs,  ['Ground Truth'] + models)):

        mult = 4 if degradation == 'id' and i>0 else 1

        tmp_crop_x, tmp_crop_y, tmp_crop_w, tmp_crop_h = mult*crop_x, mult*crop_y, mult*crop_w, mult*crop_h  # modify

        crop = sr[12][tmp_crop_y: tmp_crop_y+tmp_crop_h, tmp_crop_x: tmp_crop_x+ tmp_crop_w]
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(crop, cmap="gray")
        ax.axis("off")

        # add method name
        ax.set_title(name)

        crop = sr[7][tmp_crop_y: tmp_crop_y+tmp_crop_h, tmp_crop_x: tmp_crop_x+ tmp_crop_w]
        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(crop, cmap="gray")
        ax.axis("off")

        crop = sr[17][tmp_crop_y: tmp_crop_y+tmp_crop_h, tmp_crop_x: tmp_crop_x+ tmp_crop_w]
        ax = fig.add_subplot(gs[2, i + 1])
        ax.imshow(crop, cmap="gray")
        ax.axis("off")

    for i, wl in enumerate([500, 610, 720]):
        ax = fig.add_subplot(gs[i, -1])
        ax.text(0.5, 0.5, f'{wl} nm',
            ha='center', va='center',
            fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    path = f'result_vis/master_plots/qualitative_plot-{degradation}-{number}.pdf'
    plt.savefig(path, dpi=1200)

    return path