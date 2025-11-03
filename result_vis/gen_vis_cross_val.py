import os
import numpy as np
import pandas as pd

from get_results import parse_all
from compare_results import create_model_page
from gen_latex import wrap_in_latex
from generate_vis import create_qual

def parse_run(root):

    '''
    Input - directory containing tensorboard information for one cross validation run
    Returns - Dictionary with an entry for each hold out set
            - each entry is again a Dictionary containing SSIM, PSNR, SAM and SRE for each individual test set 
            - as well as avaraged and Train Loss
    '''

    dfs = {}

    for hold_out in os.listdir(root):
        name = hold_out.split('=')[1]
        dfs[name] = parse_all(os.path.join(root, hold_out))

    return dfs

def average_runs(dfs):

    '''
    Input - dictionary containing dictionaries containing Tensorboard statitistics for each hold out set
    Output - each statistic (PSNR, SSIM, SAM, SRE) averaged over the hold out sets
    '''

    df = {}

    for metric in ['Avg PSNR', 'Avg SSIM', 'Avg SAM', 'Avg SRE', 'Train Loss']:
        df[metric] = np.zeros(80)
        for key in list(dfs.keys()):
            if 'HCI_new' not in key:
                df[metric] += np.array(dfs[key][metric]['value'])[:80] / 4

    #convert the numpy array to a pd dataframe, so it can be processed by other functions
    for metric in ['Avg PSNR', 'Avg SSIM', 'Avg SAM', 'Avg SRE', 'Train Loss']:
        df_aux = np.concat([np.arange(80).reshape((80, 1)), df[metric].reshape((80, 1))], axis=-1)
        df[metric] = pd.DataFrame(
            df_aux, columns=['step', 'value']
        )
    
    return {'avg': df}


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='distg')
    parser.add_argument("--exp", type=str, default='run1-1028-1656')
    args = parser.parse_args()

    model=args.model
    exp_name=args.exp
    root=f'runs/{model}/cross_val/{exp_name}'
    dfs = parse_run(root)
    avg = average_runs(dfs)

    intra, _ = create_model_page(model, dfs)
    avg, _ = create_model_page('Average over Hold Out Sets', avg)
    qual = create_qual(model, 'cross_val/'+exp_name)

    info1 = 'The name indicates the training set which was removed.'
    info2 = 'Averaging the results over the hold out sets.'

    latex = wrap_in_latex(
        info1 + '\n' + intra + '\n' + '\\newpage' + '\n' + info2 + avg
        + '\n' + '\\newpage' + '\n' + qual
    )
    dir = 'result_vis/latex'
    with open(dir+f'/cross_val.tex', 'w') as fout:
        for i in range(len(latex)):
            fout.write(latex[i])