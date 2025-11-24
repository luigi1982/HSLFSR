import os
import subprocess

from get_results import parse_all
from gen_latex import wrap_in_latex

def get_stats(models, exps):

    ### load all the statistics for the specified experiments

    dic = {}
    for model, exp in zip(models, exps):
        path = os.path.join('runs', model, 'evaluate', exp)
        dfs = parse_all(path)
        dic[model] = dfs

    return dic

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


def per_view_statistics(dicts):
    print(dicts['epit'].keys())



def main():
    models = ['epit', 'det', 'f3dun']
    exps = ["real_data-run1-1031-2301", "real_data-run1-1106-1620", "real_data-run1-1107-0158"]

    for model, exp in zip(models, exps):
        check_evaluation(model, exp)

    dicts = get_stats(models, exps)

    per_view_statistics(dicts)

    return

    latex = create_metrics_table(dicts)
    latex = wrap_in_latex(latex)

    with open(f'result_vis/master/v1.tex', 'w') as fout:
        for i in range(len(latex)):
            fout.write(latex[i])

if __name__ == '__main__':
    main()
