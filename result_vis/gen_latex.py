def latex_table(rows_desc, rows, cols):
    
    latex = '\\begin{tabular}' + '{c' + len(cols)*'|c' + '} \n '
    
    for col in cols:
        if '_' in col:
            col = col.replace('_', r'\_')

        latex += ' & ' + col
    
    latex += ' \\\\ \n \\hline \n'
    
    for row, desc in zip(rows, rows_desc):
        latex += desc
        for e in row:
            latex += ' & ' + f'{e:.3f}'
        latex += ' \\\\ \n \\hline \n'

    latex += '\\end{tabular}'

    return latex

def latex_figure(caption, path_to_image):

    latex = rf'''
\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.9\textwidth]{{{path_to_image}}}
    \caption{{{caption}}}
    \label{{fig:{caption}}}
\end{{figure}}'''
    
    return latex


def double_table(dict, d1, d2):

    latex = rf'''
        \begin{{table}}[h]
        \centering
        \begin{{tabular}}{{c || c c c c || c c c c}}
            & \multicolumn{{4}}{{c||}}{{Indoors night}} 
            & \multicolumn{{4}}{{c}}{{Multi-exposure}} \\
            \cline{{2-9}}
            & PSNR & SSIM & SAM & SRE 
            & PSNR & SSIM & SAM & SRE \\
            \hline
        '''
    
    for i, model in enumerate(dict.keys()):

        psnr = f'PSNR {d1}'
        ssim = f'SSIM {d1}'
        sam = f'SAM {d1}'
        sre = f'SRE {d1}'

        latex += f'''{model} & {dict[model][psnr]['value'][0]:.2f} & {dict[model][ssim]['value'][0]:.4f} & {dict[model][sam]['value'][0]:.4f} & {dict[model][sre]['value'][0]:.2f}'''

        psnr = f'PSNR {d2}'
        ssim = f'SSIM {d2}'
        sam = f'SAM {d2}'
        sre = f'SRE {d2}'

        latex += f'''& {dict[model][psnr]['value'][0]:.2f} & {dict[model][ssim]['value'][0]:.4f} & {dict[model][sam]['value'][0]:.4f} & {dict[model][sre]['value'][0]:.2f}'''

        if i < len(dict.keys()) - 1:
            latex += '\\\\'
            latex += '\n'

    
    latex += rf'''
        \end{{tabular}}
        \caption{{
            Average PSNR, SSIM, SAM and SRE over Indoors night and Multi-exposure recordings individually.
        }}
        \label{{tab:Indoors_night-multi_exposure_rec-vanilla_bc_deg}}
    \end{{table}}'''
    
    return latex

def minipage2x2(captions, files):
    
    return fr'''\begin{{figure}}[htbp]
    \centering
    %--- Top row ---
    \begin{{minipage}}[b]{{0.45\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{files[0]}}}
        \caption*{{{captions[0]}}}
    \end{{minipage}}
    \hfill
    \begin{{minipage}}[b]{{0.45\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{files[1]}}}
        \caption*{{{captions[1]}}}
    \end{{minipage}}
    \vspace{{0.5cm}}
    %--- Bottom row ---
    \begin{{minipage}}[b]{{0.45\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{files[2]}}}
        \caption*{{{captions[2]}}}
    \end{{minipage}}
    \hfill
    \begin{{minipage}}[b]{{0.45\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{files[3]}}}
        \caption*{{{captions[3]}}}
    \end{{minipage}}
\end{{figure}}'''

def wrap_in_latex(body):
    header=r'''\documentclass{article}
\usepackage{graphicx} 
\begin{document}

'''
    footer=r'''\end{document}'''

    return header+body+footer

def gen_description(dict, num_parameters):
    latex = f'''The {dict['model']['model']} is used with a channel dimension of ${dict['model']['dim']}$.
    The model has an overall number of ${num_parameters}$ of learnable parameters.
    The model was trained for ${dict['epochs']}$ epochs with a batch size of ${dict['batch_size']}$.
    The {dict['optim']['name']} optimizer was used with an initial learning rate of ${dict['optim']['lr']}$.
    The learning rate is decreased every ${dict['optim']['lr_decay_steps']}$ epochs by a factor of ${dict['optim']['gamma']}$.'''
    return latex

def generate_minipage_figure(figures, caption="Overall caption", label="fig:multi"):
    """
    figures: list of strings (e.g. image filenames without extension handling logic)
    caption: overall figure caption
    label:   LaTeX label for \ref

    Returns a LaTeX string.
    """
    n = len(figures)
    if n == 0:
        raise ValueError("Need at least one figure")

    lines = []
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")

    # Handle pairs of figures (2 columns)
    # If n is odd, we leave the last one for separate handling.
    last_index_for_pairs = n if n % 2 == 0 else n - 1

    for i in range(0, last_index_for_pairs, 2):
        fig1 = figures[i]
        fig2 = figures[i + 1]

        lines.append(r"    %% Row for figures %d and %d" % (i + 1, i + 2))
        lines.append(r"    \begin{minipage}[b]{0.48\textwidth}")
        lines.append(r"        \centering")
        lines.append(rf"        \includegraphics[width=\textwidth]{{{fig1}}}")
        # optional: per-subfigure caption, e.g. \subcaption{...}
        # lines.append(r"        \subcaption{}")
        lines.append(r"    \end{minipage}%")
        lines.append(r"    \hfill")
        lines.append(r"    \begin{minipage}[b]{0.48\textwidth}")
        lines.append(r"        \centering")
        lines.append(rf"        \includegraphics[width=\textwidth]{{{fig2}}}")
        # lines.append(r"        \subcaption{}")
        lines.append(r"    \end{minipage}")
        lines.append(r"")
        lines.append(r"    \vspace{0.5em}")  # vertical space between rows (adjust as you like)")
        lines.append(r"")

    # If odd, handle last figure: centered horizontally
    if n % 2 == 1:
        last_fig = figures[-1]
        lines.append(r"    %% Last single figure (centered)")
        # width < \textwidth so that centering is visible; tweak 0.6 as desired
        lines.append(r"    \begin{minipage}[b]{0.6\textwidth}")
        lines.append(r"        \centering")
        lines.append(rf"        \includegraphics[width=\textwidth]{{{last_fig}}}")
        # lines.append(r"        \subcaption{}")
        lines.append(r"    \end{minipage}")
        lines.append(r"")

    lines.append(r"    \caption{" + caption + r"}")
    lines.append(r"    \label{" + label + r"}")
    lines.append(r"\end{figure}")

    return "\n".join(lines)

#print(latex_figure('PSNR', 'psnr.png'))