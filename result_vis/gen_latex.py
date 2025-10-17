def latex_table(rows_desc, rows, cols):
    
    latex = '\\begin{tabular}' + '{c' + len(cols)*'|c' + '} \n '
    
    for col in cols:
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
    \includegraphics[width=0.7\textwidth]{{{path_to_image}}}
    \caption{{{caption}}}
    \label{{fig:{caption}}}
\end{{figure}}'''
    
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
    The learning rate is decreased every ${dict['optim']['weight_decay_steps']}$ epochs by a factor of ${dict['optim']['gamma']}$.'''
    return latex

#print(latex_figure('PSNR', 'psnr.png'))