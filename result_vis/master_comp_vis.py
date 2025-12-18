import os
import subprocess
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import numpy as np
import torch
from torch.nn import functional as F

from get_results import parse_all
from gen_latex import wrap_in_latex, latex_figure, double_table, generate_minipage_figure
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

        if model == 'epit':
            model_name = 'epit_small'
        else:
            model_name = model

        dic[model_name] = dfs

        stats[model_name] = {}
        num_params, num_flops, time = get_model_stats(model)
        num_params /= 1e6
        num_flops /= 1e9
        stats[model_name]['prms'] = num_params
        stats[model_name]['flops'] = num_flops
        stats[model_name]['time'] = time

    return dic, stats

def get_training_stats(models, exps):
        
    dic = {}
    for model, exp in zip(models, exps):

        path = os.path.join('runs', model, 'training', exp)
        dfs = parse_all(path)

        if model in list(dic.keys()):
            model_name = 'epit_small'
        else:
            model_name = model

        dic[model_name] = dfs

    return dic

def multi_tloss_vs_psnr(dfs, model_names):

    figures = []

    for i, model in enumerate(list(dfs.keys())):
        path = train_loss_vs_psnr(dfs[model], model, model_names[i])
        figures.append(path)

    return generate_minipage_figure(figures)

def train_loss_vs_psnr(df, model, model_name,
                       train_ylim=0.025, psnr_ylim=41.5):

    train_loss = df['Train Loss']
    psnr = df['Avg PSNR']

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.3
    })

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # --- Train loss axis ---
    ax1.plot(train_loss['step'], train_loss['value'], label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")

    #if train_ylim is not None:
    #    ax1.set_ylim(train_ylim)

    # --- PSNR axis ---
    ax2 = ax1.twinx()
    ax2.plot(psnr['step'], psnr['value'], color='orange', label="PSNR")
    ax2.set_ylabel("PSNR (dB)")

    #if psnr_ylim is not None:
    #    ax2.set_ylim(psnr_ylim)


    # Legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.title(model_name)
    plt.tight_layout()

    path = f'result_vis/master_plots/train_loss_vs_psnr-{model_name}.pdf'
    plt.savefig(path)
    plt.close()

    return path


def get_per_view_stats(models, exps, degradation):

    psnr_pv = {}
    ssim_pv = {}
    for model, exp in zip(models, exps):
        path = os.path.join('runs', model, 'evaluate', exp, degradation, 'per_view_statistics')

        if model in list(psnr_pv.keys()):
            model_name = 'epit_small'
        else:
            model_name = model

        psnr_pv[model_name] = {}
        ssim_pv[model_name] = {}
        
        with h5py.File(path+'/PSNR_pV.h5', 'r') as f:
            for key in list(f.keys()):
                psnr_pv[model_name][key] = np.array(f[key])

        with h5py.File(path+'/SSIM_pV.h5', 'r') as f:
            for key in list(f.keys()):
                ssim_pv[model_name][key] = np.array(f[key])

    return psnr_pv, ssim_pv


def check_evaluation(model, exp):

    if not os.path.isdir(os.path.join('runs', model, 'evaluate', exp, 'bicubic')):

        #evaluate

        script_path = os.path.join(os.path.dirname(__file__), '..', 'bash', 'evaluate.sh')
        script_path = os.path.abspath(script_path)

        task = 'diff' if model == 'distg_unet' else 'lfsr'
        config = 'distg_unet' if model == 'distg_unet' else model.split('_')[0]

        subprocess.call([script_path, task, model, config, exp])

        #rename files
        
        script_path = os.path.join(os.path.dirname(__file__), '..', 'result_vis', 'fix_text_results_names.sh')
        script_path = os.path.abspath(script_path)

        subprocess.call([script_path, model, exp])


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

    stats_header_col = ' || c c c' if stats is not None else ''
    stats_header_info = '& \#Prm. & \#FlOps & time forward pass' if stats is not None else '' 

    latex = rf'''
    \begin{{center}}
        \begin{{tabular}}{{c | c c c c{stats_header_col}}}
            & PSNR & SSIM & SAM & SRE {stats_header_info}\\
            \hline'''
        
    for i, model in enumerate(dicts.keys()):
        latex += rf'''
            {model} & {dicts[model][psnr]['value'][0]:.2f} & {dicts[model][ssim]['value'][0]:.4f} & {dicts[model][sam]['value'][0]:.4f} & {dicts[model][sre]['value'][0]:.2f}'''
        
        if stats is not None:
            latex += rf'''& {stats[model]['prms']:.2f} & {stats[model]['flops']:.2f} & {stats[model]['time']:.2f}'''
        
        if i < len(dicts.keys()) - 1:
            latex += '\\\\'

    latex += r'''
        \end{tabular}
    \end{center}
    '''
    
    return latex


def per_view_statistics(psnr, degradation, model_names):

    dataset = 'Indoors_night'
    indoors_night = per_view_statistics_dataset({model : psnr[model][dataset] for model in list(psnr.keys())}, dataset, degradation, model_names)

    dataset = 'multi_exposure_rec'
    multi_view_rec = per_view_statistics_dataset({model : psnr[model][dataset] for model in list(psnr.keys())}, dataset, degradation, model_names)

    latex = latex_figure('Average PSNR per view over the Indoors night dataset', indoors_night)
    latex += '\n \n'
    latex += latex_figure('Average PSNR per view over the Multi Exposure Rec dataset', multi_view_rec) 

    return latex


def per_view_statistics_dataset(stats, dataset, degradation, model_names):
    
    # stats: dict
    # key model/ value stat
    # pass stats (PSNR/ SSIM) for all models and one dataset in variable stats

    # display the stat per view for each model in one row

    n = len(list(stats.keys()))
    gs = GridSpec(1, n)
    fig = plt.figure(figsize=(14, 10))

    vmin = min([stats[model].min() for model in list(stats.keys())])
    vmax = max([stats[model].max() for model in list(stats.keys())])

    for i, model in enumerate(list(stats.keys())):

        img = stats[model]

        ax = fig.add_subplot(gs[i])
        im = ax.imshow(img, cmap="Reds", vmin=vmin, vmax=vmax)

        # annotate cells
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                ax.text(c, r, f"{img[r,c]:.2f}",
                        ha="center", va="center", fontsize=8)
                
        ax.set_title(model_names[i])

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

    path = os.path.join('result_vis', 'master_plots', f'{dataset}-{degradation}-psnr_per_view.pdf')
    plt.savefig(path, dpi=1200)
    plt.clf()

    return path


def qualitative_results(models, exps, degradation, model_names, datasets=None, num_scenes=1, custom_box_coords=None):

    if datasets is None:
        datasets=['Lab_day', 'Lab_night', 'Indoors_day', 'Indoors_night', 'Showroom', 'Outdoors', 'multi_exposure_rec']

        captions = [
            f'Qualitative results for dataset {' '.join(dataset.split('_'))}' for dataset in datasets
        ]

    else:
        captions = None

    hr_paths = sorted(get_lfhr_paths(datasets=datasets, num_scenes=num_scenes))

    sr_paths = []

    for model, exp in zip(models, exps):
        sr_paths.append(sorted(get_lfsr_paths(model, exp, degradation, datasets=datasets, num_scenes=num_scenes)))

    latex = r'''\begin{figure}[h!]
    \centering'''

    if captions is None:
        captions = len(sr_paths[0]) * ['Caption']

    for i in range(len(sr_paths[0])):

        if custom_box_coords is not None:
            box = custom_box_coords[i]
        else:
            box = None

        path = create_single_qual_plot(hr_paths[i], [paths[i] for paths in sr_paths], models, i, degradation, model_names, custom_box_coords=box)
        latex += rf'''
        \begin{{subfigure}}{{0.9\textwidth}}
                \centering
                \includegraphics[width=\textwidth]{{{path}}}
                \caption{{{captions[i]}}}
            \end{{subfigure}}\par\medskip
        '''
        latex += '\n'

    latex += r'''\end{figure}'''
    
    return latex


def main_evaluation():

    models = ['bicubic', 'epit', 'f3dun', 'distg_unet']
    model_names = ['Bicubic', 'EPIT', 'F3DUN', 'Distg UNet']
    exps = ["none", "new_arrangement-1205-1108", "new_arrangement-1203-2140", "new_arrangement-1210-1835"]

    for model, exp in zip(models, exps):
        check_evaluation(model, exp)

    ### bicubic downsampling

    #get the stats
    dicts, stats = get_stats(models, exps, 'bicubic')
    psnr_pv, ssim_pv = get_per_view_stats(models, exps, 'bicubic')
    intra_training_stats = get_training_stats(models[1:], exps[1:])

    #quantitative
    latex = create_metrics_table(dicts, dataset='Avg', stats=stats)
    latex += double_table(dicts, 'Indoors night', 'multi exposure rec')
    latex += per_view_statistics(psnr_pv, 'bicubic', model_names)

    #loss vs psnr
    latex += multi_tloss_vs_psnr(intra_training_stats, model_names[1:])

    #qualitative
    latex += qualitative_results(models, exps, 'bicubic', model_names,
                                 custom_box_coords=[(110, 40, 80, 80)]+2*[None] + [(60, 260, 80, 80), (260, 80, 80, 80)] + [None] + [(120, 100, 80, 80)]
    )

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
    latex += double_table(dicts, 'Indoors night', 'multi exposure rec')
    latex += per_view_statistics(psnr_pv, 'classical', model_names)

    #qualitative
    latex += qualitative_results(models, exps, 'classical', model_names,
                                custom_box_coords=[(110, 40, 80, 80)]+2*[None] + [(60, 260, 80, 80), (260, 80, 80, 80)] + [None] + [(120, 100, 80, 80)]
    )

    latex += rf'''
        \newpage
        \subsubsection{{Direct Upsampling}}
    '''

    ### no degradation
    latex += qualitative_results(models, exps, 'id', model_names, datasets=['Texts'], num_scenes=3, 
                                 custom_box_coords=[(360, 0, 50, 50), (100, 10, 120, 120), (240, 280, 100, 100)])

    return latex

def epit_evaluation():

    models = ['epit', 'epit', 'epit_short', 'epit_swin_v1', 'epit_win']
    model_names = ['EPIT', 'EPIT rcd', 'EPIT rbn', 'EPIT SWin', 'EPIT Win']
    exps = ["new_arrangement-1205-1108", "new_arrangement-1204-1723", "3-Blocks-1211-1124", "5-Blocks-1210-1629", "5-Blocks-1211-0058"]

    for model, exp in zip(models, exps):
        check_evaluation(model, exp)

    ### bicubic downsampling

    #get the stats
    dicts, stats = get_stats(models, exps, 'bicubic')
    psnr_pv, ssim_pv = get_per_view_stats(models, exps, 'bicubic')
    intra_training_stats = get_training_stats(models, exps)

    #quantitative
    latex = create_metrics_table(dicts, dataset='Avg', stats=stats)
    latex += double_table(dicts, 'Indoors night', 'multi exposure rec')
    latex += per_view_statistics(psnr_pv, 'bicubic', model_names)

    #loss vs psnr
    latex += multi_tloss_vs_psnr(intra_training_stats, model_names)

    #qualitative
    latex += qualitative_results(models, exps, 'bicubic', model_names,
                                custom_box_coords=[(110, 40, 80, 80)]+2*[None] + [(60, 260, 80, 80), (260, 80, 80, 80)] + [None] + [(120, 100, 80, 80)],                       
    )

    return latex


def appendix_evaluation():
    models = ['det', 'drcan', 'swinir', 'hat']
    model_names = ['DET', 'DRCAN', 'SWinIR', 'HAT']
    exps = ['new_arrangement-1201-1243', 'real_data-run1-1103-1722', 'real_data-run1-1104-0852', 'real_data-run1-1105-1221']

    ### bicubic downsampling

    #get the stats
    dicts, stats = get_stats(models, exps, 'bicubic')
    intra_training_stats = get_training_stats(models, exps)
    psnr_pv, ssim_pv = get_per_view_stats(models, exps, 'bicubic')

    #quantitative
    latex = create_metrics_table(dicts, dataset='Avg', stats=stats)
    latex += double_table(dicts, 'Indoors night', 'multi exposure rec')
    latex += per_view_statistics(psnr_pv, 'bicubic', model_names)

    #loss vs psnr
    latex += multi_tloss_vs_psnr(intra_training_stats, model_names)

    #qualitative
    latex += qualitative_results(models, exps, 'bicubic', model_names,
                                    custom_box_coords=[(110, 40, 80, 80)]+2*[None] + [(60, 260, 80, 80), (260, 80, 80, 80)] + [None] + [(120, 100, 80, 80)]
    )

    return latex

def diff_evaluation():

    models = ['distg_unet', 'epit', 'distg', 'distg_unet_reg']
    model_names = ['Distg UNet', 'EPIT rcd', 'Distg', 'Distg UNet reg']
    exps = ["new_arrangement-1210-1835", "new_arrangement-1204-1723", "new_arrangement-1203-1820", "new_arrangement-1210-1556"]

    for model, exp in zip(models, exps):
        check_evaluation(model, exp)

    ### bicubic downsampling

    #get the stats
    dicts, stats = get_stats(models, exps, 'bicubic')
    intra_training_stats = get_training_stats(models, exps)
    psnr_pv, ssim_pv = get_per_view_stats(models, exps, 'bicubic')

    #quantitative
    latex = create_metrics_table(dicts, dataset='Avg', stats=stats)
    latex += double_table(dicts, 'Indoors night', 'multi exposure rec')
    latex += per_view_statistics(psnr_pv, 'bicubic', model_names)

    #loss vs psnr
    latex += multi_tloss_vs_psnr(intra_training_stats, model_names)

    #qualitative
    latex += qualitative_results(models, exps, 'bicubic', model_names,
                                    custom_box_coords=[(110, 40, 80, 80)]+2*[None] + [(60, 260, 80, 80), (260, 80, 80, 80)] + [None] + [(120, 100, 80, 80)]
    )

    latex += rf'''
        \newpage
        \subsubsection{{Further Degradation}}
    '''

    return latex


def main():

    latex = appendix_evaluation()
    latex = wrap_in_latex(latex)

    with open(f'result_vis/master/appendix_v1.tex', 'w') as fout:
        for i in range(len(latex)):
            fout.write(latex[i])

if __name__ == '__main__':
    main()
