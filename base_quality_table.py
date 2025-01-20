import numpy as np
from matplotlib.colors import ListedColormap
from lm_polygraph.utils.manager import UEManager
import tabulate
import pandas as pd
from functools import partial
import seaborn as sns
import re
import matplotlib.pyplot as plt
import argparse

cm = sns.color_palette("coolwarm", as_cmap=True)

models = ['mistral7b', 'llama8b', 'falcon7b']
metrics = ['AlignScoreOutputTarget',
           'AlignScoreOutputTarget',
           'Accuracy',
           'Accuracy',
           'Rouge_rougeL',
           'Comet',
           'Comet']

datasets = [
    'trivia',
    'mmlu',
    'coqa',
    'gsm8k_cot',
    'xsum',
    'wmt14_fren',
    'wmt19_deen',
]

column_names = {
    'Greedy': '',
    'First Sample': 'Sample',
    'Best Sample': 'BestSample',
    'Best Normalized Sample': 'BestNormalizedSample',
}

def postprocess_latex(latex):
    latex = latex.replace('_', '\_')
    latex = latex.replace('table', 'table*')
    latex = latex.splitlines()

    new_latex = []
    for i, line in enumerate(latex):
        if i == 0:
            new_latex.append(line)
            new_latex.append('\\footnotesize')
        elif line.startswith('\\begin{tabular}'):
            new_latex.append('\\begin{tabular}{|l|c|c|c|c|}')
            new_latex.append('\\hline')
        elif line.endswith('\\\\'):
            new_latex.append(line + ' \\hline')
        else:
            new_latex.append(line)

    latex = '\n'.join(new_latex)

    return latex

for model in models:
    base_dir = 'sample_metric_mans/best_sample_with_greedy_enriched'

    rows = {}
    for metric_name, dataset in zip(metrics, datasets):
        man = UEManager.load(f'{base_dir}/{model}_{dataset}.man')

        row = []
        for column_name, metric_prefix in column_names.items():
            cur_metric_name = metric_prefix + metric_name
            metric_values = man.gen_metrics[('sequence', cur_metric_name)]
            row.append(np.mean(metric_values))
        rows[dataset] = row

    df = pd.DataFrame.from_dict(rows, orient='index', columns=column_names.keys())
    latex = df.style.format(precision=3).set_caption(f'Base quality metrics for {model}').to_latex()
    latex = postprocess_latex(latex)

    with open(f'{base_dir}/{model}_base_quality.tex', 'w') as f:
        f.write(latex)
