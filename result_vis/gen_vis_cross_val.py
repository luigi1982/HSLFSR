import os

from get_results import parse_all
from compare_results import create_model_page
from gen_latex import wrap_in_latex

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

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='epit')
    parser.add_argument("--exp", type=str, default='normalization-1027-1319')
    args = parser.parse_args()

    model=args.model
    exp_name=args.exp
    root=f'runs/{model}/cross_val/{exp_name}'
    dfs = parse_run(root)

    intra, _ = create_model_page(model, dfs)

    info = 'The name indicates the training set which was removed.'

    latex = wrap_in_latex(
        info + '\n' + intra
    )
    dir = 'result_vis/latex'
    with open(dir+f'/cross_val.tex', 'w') as fout:
        for i in range(len(latex)):
            fout.write(latex[i])