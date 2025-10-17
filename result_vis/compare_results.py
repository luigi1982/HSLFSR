### Compile the results, avarage PSNR SSIM SAM SRE into one file
### First one chapter per model comparing different experiments
### One last chapter comparing the top models -> which metric?

import os
from matplotlib import pyplot as plt
import numpy as np

from get_results import parse_avgs
from generate_vis import gen_table
from gen_latex import minipage2x2, wrap_in_latex, latex_figure

def get_stats():
    dir = 'runs'
    dic = {}
    for model in os.listdir(dir):
        dic[model] = {}
        path = os.path.join(dir, model, 'training')
        for exp in os.listdir(path):
            if not 'no-normalization' in exp:
                try: 
                    dfs = parse_avgs(os.path.join(path, exp))
                    if len(dfs) >= 4:
                        dic[model][exp] = dfs
                except:
                    pass

    return dic


def plot_dfs(model, metric, dfs, labels, title, y_label, x_label='epochs'):
    for df, label in zip(dfs, labels):
        x = df['step'].values
        y = df['value'].values
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    dir = 'result_vis/plots/compare/' + model
    os.makedirs(dir, exist_ok=True)
    file=dir+f'/{metric}.pdf'
    plt.savefig(file)
    plt.close()

    return file


def create_model_page(model, model_dict):

    ### plot the loss
    if model != 'bicubic':
        dfs = [model_dict[exp]['Train Loss'] for exp in model_dict.keys()]
        train_loss_plot = plot_dfs(model, 'Train-Loss', dfs, list(model_dict.keys()), 'Train Loss', 'L1 Loss')

    ### plot metrics
    metric_plots = []
    for metric in ['PSNR', 'SSIM', 'SAM', 'SRE']:
        dfs = [model_dict[exp]['Avg '+metric] for exp in model_dict.keys()]
        metric_plots.append(plot_dfs(model, metric, dfs, list(model_dict.keys()), metric, metric))

    ### create tables
    metric_tables = []
    for metric in ['PSNR', 'SSIM', 'SAM', 'SRE']:
        rows_desc = ['MAX' if metric in ['PSNR', 'SSIM'] else 'MIN', 'LAST']
        dfs = [model_dict[exp]['Avg '+metric] for exp in model_dict.keys()]
        labels = list(model_dict.keys())
        take_max = metric in ['PSNR', 'SSIM']
        metric_tables.append(gen_table(rows_desc, dfs, cols_desc=labels, take_max=take_max))

    ### get the best model in terms of PSNR
    psnr = [model_dict[exp]['Avg PSNR'] for exp in model_dict.keys()]
    psnr_max = np.array([df['value'].max() for df in psnr])
    index = np.argmax(psnr_max)
    exp_max = model_dict[list(model_dict.keys())[index]]

    ### assamble latex
    ### Train Loss -> Metric Plots -> Metric Tables

    latex = f'\\subsection{{{model}}}'

    if model != 'bicubic':
        latex += latex_figure('Train Loss', train_loss_plot)

    latex += minipage2x2(['PSNR', 'SSIM', 'SAM', 'SRE'], metric_plots)
    
    for metric, t in zip(['PSNR', 'SSIM', 'SAM', 'SRE'], metric_tables):
        latex += f'\\textbf{{{metric}}} \\newline'
        latex += t
        latex += '\\newline'
    
    return latex, exp_max

def create_model_pages(dic):

    latex = '\\section{Intra Model Comparison}'
    max_models = {}

    dir = 'runs'
    for model in os.listdir(dir):
        if model != 'distg_unet':
            latex_model, max_model = create_model_page(model, dic[model])
            max_models[model] = max_model
            latex += latex_model

    return latex, max_models


def create_comparison(max_models):

    ### plot the loss
    dfs = [max_models[model]['Train Loss'] for model in max_models.keys() if model != 'bicubic']
    train_loss_plot = plot_dfs('Loss-Comparison', 'Train-Loss', dfs, list(max_models.keys()), 'Train Loss', 'L1 Loss')

    ### plot metrics
    metric_plots = []
    for metric in ['PSNR', 'SSIM', 'SAM', 'SRE']:
        dfs = [max_models[model]['Avg '+metric] for model in max_models.keys() if model != 'bicubic']
        metric_plots.append(plot_dfs(metric+'-Comparison', metric, dfs, list(max_models.keys()), metric, metric))

    ### create tables
    metric_tables = []
    for metric in ['PSNR', 'SSIM', 'SAM', 'SRE']:
        rows_desc = ['MAX' if metric in ['PSNR', 'SSIM'] else 'MIN', 'LAST']
        dfs = [max_models[model]['Avg '+metric] for model in max_models.keys()]
        labels = list(max_models.keys())
        take_max = metric in ['PSNR', 'SSIM']
        metric_tables.append(gen_table(rows_desc, dfs, cols_desc=labels, take_max=take_max))

    ### assamble latex
    ### Train Loss -> Metric Plots -> Metric Tables

    latex = '\\section{Inter Model Comparison}'

    latex += latex_figure('Train Loss', train_loss_plot)

    latex += minipage2x2(['PSNR', 'SSIM', 'SAM', 'SRE'], metric_plots)
    
    for metric, t in zip(['PSNR', 'SSIM', 'SAM', 'SRE'], metric_tables):
        latex += f'\\newline \n \\vspace{{0.25cm}} \\textbf{{{metric}}} \\vspace{{0.25cm}} \\newline'
        latex += t

    latex += '\\newpage'
    
    return latex

def create_doc():
    dic = dict([(k, v) for k, v in get_stats().items() if k != 'distg_unet'])
    intra, models = create_model_pages(dic)
    inter = create_comparison(models)
    latex = wrap_in_latex(
        '\\title{Intra and Inter Model Comparison} \n \\maketitle \\tableofcontents \n' + intra + inter
    )
    dir = 'result_vis/latex'
    with open(dir+f'/compare.tex', 'w') as fout:
        for i in range(len(latex)):
            fout.write(latex[i])


if __name__ == '__main__':
    create_doc()