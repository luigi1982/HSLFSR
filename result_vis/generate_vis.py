from matplotlib import pyplot as plt
import pandas as pd
import os
import yaml
import sys

from get_results import parse_all
from gen_latex import *
from get_lightfields import create_qual_plot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import MODEL_REGISTRY

### Creae plots and tables for mterics
### Quantitative Results

def plot_dfs(exp_name, metric, dfs, labels, title, y_label, x_label='epochs'):
    for df, label in zip(dfs, labels):
        x = df['step'].values
        y = df['value'].values
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    dir = 'result_vis/plots/'+exp_name
    os.makedirs(dir, exist_ok=True)
    file=dir+f'/{metric}.pdf'
    plt.savefig(file)
    plt.close()

    return file

def gen_table(rows_desc, dfs, cols_desc, take_max):

    rows = [[], []]
    for df in dfs:
        m = df['value'].max() if take_max else df['value'].min()
        l = df['value'].values[-1]
        rows[0].append(m)
        rows[1].append(l)

    return latex_table(rows_desc, rows, cols_desc)

def create_metric_page(exp_name, metric, labels, dfs, take_max):

    #create plot
    file = plot_dfs(exp_name, metric, dfs, labels, title=metric, y_label=metric, x_label='epochs')
    #create latex yable
    rows_desc = ['MAX' if take_max else 'MIN', 'LAST']
    table = gen_table(rows_desc, dfs, cols_desc=labels, take_max=take_max)

    return file,table

def create_metric_pages(exp_name, dfs, fraunhofer_available=True):

    metrics= ['PSNR', 'SSIM', 'SAM', 'SRE']
    if fraunhofer_available:
        labels=['EPFL', 'HCI new', 'HCI old', 'INRIA Lytro', 'Stanford Gantry', 'Fraunhofer']
    else:
        labels=['EPFL', 'HCI new', 'HCI old', 'INRIA Lytro', 'Stanford Gantry']

    latex = '\\section{Quantitative Results} \n'

    for metric, take_max in zip(metrics, [True, True, False, False]):

        filtered_dfs=[]
        for label in labels:
            label = metric+' '+label
            filtered_dfs.append(dfs[label])

        #create plot and table
        file, table = create_metric_page(exp_name, metric, labels, filtered_dfs, take_max=take_max)
        #create figure with the plot
        figure = latex_figure(metric, file)
        #combine figure and table
        latex += f'\\subsection{{{metric}}} \n'
        latex += figure + '\n\n' + table + '\n\n'

    return latex

### Create plots of the images
### Qualitative Results

def create_qual(model, exp_name):
    latex = '\\section{Qualitative Results} \n'
    file=create_qual_plot(model, exp_name)
    latex+=latex_figure(
        'Qualitative Results, from left to right, Bicubically upsampled, Super Resolved, Original Data',
        file
    )
    return latex

### Create a Describtion of the models

def create_model_desc(model, exp_name, df):
    dir=os.path.join('results', model, exp_name)
    if os.path.exists(dir+'/config.yaml'):
        file=os.path.join(dir+'/config.yaml')
    else:
        file = os.path.join('configs', model+'.yaml')
    
    with open(file, 'r') as f:
        cfg = yaml.safe_load(f)

    model_class = MODEL_REGISTRY[cfg['train']['model']['model']]
    net = model_class(cfg['train']['model']['dim'])
    num_parameters=sum(p.numel() for p in net.parameters())

    latex = '\\section{Model Description and Training} \n'
    latex += gen_description(cfg['train'], num_parameters)
    file = plot_dfs(exp_name, 'train-loss', [df], ['L1 Loss'], 'Train Loss', 'L1 Loss', x_label='epochs')
    latex += '\n\n'
    latex += latex_figure('Train Loss', file)

    return latex

def create_documentation(model, exp_name, dfs, fraunhofer_available=True):

    title= rf'''\title{{Model {model}, Experiment {exp_name}}}
    \maketitle
    '''
    model_desc=create_model_desc(model, exp_name, dfs['Train Loss'])
    qual=create_qual(model, exp_name)
    quant=create_metric_pages(exp_name, dfs, fraunhofer_available=fraunhofer_available)
    body=title+model_desc+'\\newpage'+qual+'\\newpage'+quant
    latex=wrap_in_latex(body)
    dir=f'result_vis/latex/{model}'
    os.makedirs(dir, exist_ok=True)
    with open(dir+f'/{model}-{exp_name}.tex', 'w') as fout:
        for i in range(len(latex)):
            fout.write(latex[i])


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='epit')
    parser.add_argument("--exp", type=str, default='normalization-0924-1031')
    parser.add_argument("--fh", type=bool, default=True)
    args = parser.parse_args()

    model=args.model
    exp_name=args.exp
    root=f'runs/{model}/training/{exp_name}'
    dfs = parse_all(root)
    create_documentation(model, exp_name, dfs, fraunhofer_available=args.fh)

