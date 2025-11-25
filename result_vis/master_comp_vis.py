import os
import subprocess
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import numpy as np
import torch
from torch.nn import functional as F

from get_results import parse_all
from gen_latex import wrap_in_latex, latex_figure
from get_lightfields import get_lfhr_paths, get_lfsr_paths
from master_qual_plots import create_single_qual_plot


def get_stats(models, exps):

    ### load all the statistics for the specified experiments

    dic = {}
    for model, exp in zip(models, exps):
        path = os.path.join('runs', model, 'evaluate', exp)
        dfs = parse_all(path)
        dic[model] = dfs

    return dic

def get_per_view_stats(models, exps):

    psnr_pv = {}
    ssim_pv = {}
    for model, exp in zip(models, exps):
        path = os.path.join('runs', model, 'evaluate', exp, 'per_view_statistics')
        psnr_pv[model] = {}
        ssim_pv[model] = {}
        
        with h5py.File(path+'/PSNR_pV.h5', 'r') as f:
            for key in list(f.keys()):
                psnr_pv[model][key] = np.array(f[key])

        with h5py.File(path+'/SSIM_pV.h5', 'r') as f:
            for key in list(f.keys()):
                ssim_pv[model][key] = np.array(f[key])

    return psnr_pv, ssim_pv


def check_evaluation(model, exp):
    if not os.path.isdir(os.path.join('runs', model, 'evaluate', exp)):
        script_path = os.path.join(os.path.dirname(__file__), '..', 'bash', 'evaluate_lfsr.sh')
        script_path = os.path.abspath(script_path)
        subprocess.call([script_path, model, exp])

def create_metrics_table(dicts):

    #create a table with a column for each model in dicts
    #and a column for every Metric, i.e. PSNR, SSIM, SAM and SRE

    latex = r'''
    \begin{center}
        \begin{tabular}{c | c c c c}
            & PSNR & SSIM & SAM & SRE \\
            \hline'''
        
    for i, model in enumerate(dicts.keys()):
        latex += rf'''
            {model} & {dicts[model]['Avg PSNR']['value'][0]:.2f} & {dicts[model]['Avg SSIM']['value'][0]:.4f} & {dicts[model]['Avg SAM']['value'][0]:.4f} & {dicts[model]['Avg SRE']['value'][0]:.2f}'''
        
        if i < len(dicts.keys()) - 1:
            latex += '\\\\'

    latex += r'''
        \end{tabular}
    \end{center}
    '''
    
    return latex


def per_view_statistics(psnr, ssim):

    dataset = 'Indoors_night'
    indoors_night = per_view_statistics_dataset({model : psnr[model][dataset] for model in list(psnr.keys())}, 'PSNR', dataset)

    dataset = 'multi_exposure_rec'
    multi_view_rec = per_view_statistics_dataset({model : psnr[model][dataset] for model in list(psnr.keys())}, 'PSNR', dataset)

    latex = latex_figure('Average PSNR per view over the Indoors night dataset', indoors_night)
    latex += '\n \n'
    latex += latex_figure('Average PSNR per view over the Multi Exposure Rec dataset', multi_view_rec) 

    return latex


def per_view_statistics_dataset(stats, metric, dataset):
    
    # stats: dict
    # key model/ value stat
    # pass stats (PSNR/ SSIM) for all models and one dataset in variable stats

    # display the stat per view for each model in one row

    n = len(list(stats.keys()))
    gs = GridSpec(1, n)
    fig = plt.figure(figsize=(14, 10))


    for i, model in enumerate(list(stats.keys())):

        img = stats[model]

        ax = fig.add_subplot(gs[i])
        im = ax.imshow(img, cmap="Greens")

        # annotate cells
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                ax.text(c, r, f"{img[r,c]:.2f}",
                        ha="center", va="center", fontsize=8)

        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Shared colorbar ---
    cbar = fig.colorbar(im,
                    ax=fig.get_axes(),
                    shrink=0.4,
                    fraction=0.03,   # width of colorbar relative to axes
                    pad=0.04) 
    cbar.ax.tick_params(labelsize=12)

    path = os.path.join('result_vis', 'master_plots', f'{dataset}-psnr_per_view.pdf')
    plt.savefig(path, dpi=1200)
    plt.clf()

    return path


def qualitative_results(models, exps):

    hr_paths = sorted(get_lfhr_paths())

    print(hr_paths)

    sr_paths = []

    for model, exp in zip(models, exps):
        sr_paths.append(sorted(get_lfsr_paths(model, exp)))

    latex = ''

    for i in range(7):

        path = create_single_qual_plot(hr_paths[i], [paths[i] for paths in sr_paths], models, i)
        latex += latex_figure('caption', path)
        latex += '\n'
    
    return latex


def main():
    models = ['epit', 'det', 'f3dun']
    exps = ["real_data-run1-1031-2301", "real_data-run1-1106-1620", "real_data-run1-1107-0158"]

    for model, exp in zip(models, exps):
        check_evaluation(model, exp)

    dicts = get_stats(models, exps)
    psnr_pv, ssim_pv = get_per_view_stats(models, exps)

    latex = create_metrics_table(dicts)
    latex += per_view_statistics(psnr_pv, ssim_pv)
    latex += qualitative_results(models, exps)
    latex = wrap_in_latex(latex)

    with open(f'result_vis/master/v1.tex', 'w') as fout:
        for i in range(len(latex)):
            fout.write(latex[i])

if __name__ == '__main__':
    main()
