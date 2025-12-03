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
from get_model_stats import get_model_stats


def get_stats(models, exps, degradation):

    ### load all the statistics for the specified experiments

    dic = {}
    stats = {}
    for model, exp in zip(models, exps):
        path = os.path.join('runs', model, 'evaluate', exp, degradation)
        dfs = parse_all(path)
        dic[model] = dfs

        stats[model] = {}
        num_params, num_flops = get_model_stats(model)
        num_params /= 1e6
        num_flops /= 1e9
        stats[model]['prms'] = num_params
        stats[model]['flops'] = num_flops

    return dic, stats

def get_per_view_stats(models, exps, degradation):

    psnr_pv = {}
    ssim_pv = {}
    for model, exp in zip(models, exps):
        path = os.path.join('runs', model, 'evaluate', exp, degradation, 'per_view_statistics')
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

    if not os.path.isdir(os.path.join('runs', model, 'evaluate', exp, 'bicubic')):

        script_path = os.path.join(os.path.dirname(__file__), '..', 'bash', 'evaluate.sh')
        script_path = os.path.abspath(script_path)

        task = 'diff' if model == 'distg_unet' else 'lfsr'

        subprocess.call([script_path, task, model, exp])


def create_metrics_table(dicts, dataset, stats=None):

    #create a table with a column for each model in dicts
    #and a column for every Metric, i.e. PSNR, SSIM, SAM and SRE

    if dataset == 'Avg':
        psnr = 'Avg PSNR'
        ssim = 'Avg SSIM'
        sam = 'Avg SAM'
        sre = 'Avg SRE'
    else:
        psnr = f'PSNR {dataset}'
        ssim = f'SSIM {dataset}'
        sam = f'SAM {dataset}'
        sre = f'SRE {dataset}'

    stats_header_col = ' || c c' if stats is not None else ''
    stats_header_info = '& \#Prm. & \#FlOps' if stats is not None else '' 

    latex = rf'''
    \begin{{center}}
        \begin{{tabular}}{{c | c c c c{stats_header_col}}}
            & PSNR & SSIM & SAM & SRE {stats_header_info}\\
            \hline'''
        
    for i, model in enumerate(dicts.keys()):
        latex += rf'''
            {model} & {dicts[model][psnr]['value'][0]:.2f} & {dicts[model][ssim]['value'][0]:.4f} & {dicts[model][sam]['value'][0]:.4f} & {dicts[model][sre]['value'][0]:.2f}'''
        
        if stats is not None:
            latex += rf'''& {stats[model]['prms']:.2f} & {stats[model]['flops']:.2f}'''
        
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


def qualitative_results(models, exps, degradation, datasets=None, num_scenes=1, custom_box_coords=None):

    if datasets is None:
        datasets=['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors', 'multi_exposure_rec']

    hr_paths = sorted(get_lfhr_paths(datasets=datasets, num_scenes=num_scenes))

    sr_paths = []

    for model, exp in zip(models, exps):
        sr_paths.append(sorted(get_lfsr_paths(model, exp, degradation, datasets=datasets, num_scenes=num_scenes)))

    latex = ''

    if num_scenes == 3:
        print(hr_paths)

    print('length HR paths: ', len(hr_paths))
    print('length SR paths: ', len(sr_paths[0]))

    for i in range(len(sr_paths[0])):

        if custom_box_coords is not None:
            box = custom_box_coords[i]
        else:
            box = None

        path = create_single_qual_plot(hr_paths[i], [paths[i] for paths in sr_paths], models, i, degradation, custom_box_coords=box)
        latex += latex_figure('caption', path)
        latex += '\n'
    
    return latex


def main():
    models = ['bicubic', 'epit', 'det', 'f3dun', 'distg_unet']
    exps = ["none", "new_arrangement-1128-1308", "real_data-run1-1106-1620", "real_data-run1-1107-0158", "new_arrangement-1128-1512"]

    for model, exp in zip(models, exps):
        check_evaluation(model, exp)

    ### bicubic downsampling

    #get the stats
    dicts, stats = get_stats(models, exps, 'bicubic')
    psnr_pv, ssim_pv = get_per_view_stats(models, exps, 'bicubic')

    #quantitative
    latex = create_metrics_table(dicts, dataset='Avg', stats=stats)
    latex += create_metrics_table(dicts, dataset='Indoors night')
    latex += create_metrics_table(dicts, dataset='multi exposure rec')
    latex += per_view_statistics(psnr_pv, ssim_pv)

    #qualitative
    latex += qualitative_results(models, exps, 'bicubic')

    latex += rf'''
        \newpage
        \subsubsection{{Further Degradation}}
    '''

    ### classical degradation

    #get the stats
    dicts, stats = get_stats(models, exps, 'classical')
    psnr_pv, ssim_pv = get_per_view_stats(models, exps, 'bicubic')

    #quantitative
    latex += create_metrics_table(dicts, dataset='Avg')
    latex += create_metrics_table(dicts, dataset='Indoors night')
    latex += create_metrics_table(dicts, dataset='multi exposure rec')
    latex += per_view_statistics(psnr_pv, ssim_pv)

    #qualitative
    latex += qualitative_results(models, exps, 'classical')

    latex += rf'''
        \newpage
        \subsubsection{{Direct Upsampling}}
    '''

    ### no degradation
    latex += qualitative_results(models, exps, 'id', datasets=['Texts'], num_scenes=3,
                                 custom_box_coords=[(240, 280, 100, 100), (100, 10, 120, 120), (360, 0, 50, 50)])

    latex = wrap_in_latex(latex)

    with open(f'result_vis/master/v1.tex', 'w') as fout:
        for i in range(len(latex)):
            fout.write(latex[i])

if __name__ == '__main__':
    main()
