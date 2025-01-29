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
from pathlib import Path
from collections import defaultdict

cm = sns.color_palette("coolwarm", as_cmap=True)

models = ['mistral7b', 'llama8b', 'falcon7b']
datasets = ['trivia', 'mmlu', 'coqa', 'gsm8k_cot', 'xsum', 'wmt14_fren', 'wmt19_deen']
sim_mat_types = ['align', 'rouge', 'nli', 'ce']

def get_metrics(args):
    if args.do_sample:
        if args.sample_strategy == 'first':
            ats_metrics = ['SampleAlignScoreInputOutput']
            nmt_metrics = ['SampleComet']
            short_qa_metrics = ['SampleAccuracy']
            long_qa_metrics = ['SampleAlignScoreOutputTarget']
        elif args.sample_strategy == 'best':
            ats_metrics = ['BestSampleAlignScoreInputOutput']
            nmt_metrics = ['BestSampleComet']
            short_qa_metrics = ['BestSampleAccuracy']
            long_qa_metrics = ['BestSampleAlignScoreOutputTarget']
        elif args.sample_strategy == 'best_normalized':
            ats_metrics = ['BestNormalizedSampleAlignScoreInputOutput']
            nmt_metrics = ['BestNormalizedSampleComet']
            short_qa_metrics = ['BestNormalizedSampleAccuracy']
            long_qa_metrics = ['BestNormalizedSampleAlignScoreOutputTarget']
        else:
            raise ValueError(f'Invalid sample strategy: {args.sample_strategy}')
    else:
        ats_metrics = ['AlignScoreInputOutput']
        nmt_metrics = ['Comet']
        short_qa_metrics = ['Accuracy']
        long_qa_metrics = ['AlignScoreOutputTarget']

    return ats_metrics, nmt_metrics, short_qa_metrics, long_qa_metrics

method_names = {
    'GreedySemanticEnrichedMaxprobAveDissimilarity': '$\\text{CoCoA}_{MSP}$',
    'GreedySemanticEnrichedPPLAveDissimilarity': '$\\text{CoCoA}_{PPL}$',
    'GreedySemanticEnrichedMTEAveDissimilarity': '$\\text{CoCoA}_{MTE}$',
    'BestSemanticEnrichedMaxprobAveDissimilarity': '$\\text{CoCoA}_{MSP}$',
    'BestSemanticEnrichedPPLAveDissimilarity': '$\\text{CoCoA}_{PPL}$',
    'BestSemanticEnrichedMTEAveDissimilarity': '$\\text{CoCoA}_{MTE}$',
}

model_names = {
    'mistral7b': 'Mistral7b-Base',
    'llama8b': 'Llama8b-Base',
    'falcon7b': 'Falcon7b-Base',
}

sim_mat_names = {
    'align': 'AlignScore',
    'rouge': 'RougeL',
    'nli': 'NLI',
    'ce': 'CrossEncoder',
}

def get_methods(args):
    single_sequence_methods = []
    focused_sample_methods = [
        'SemanticEnrichedMaxprobAveDissimilarity',
        'SemanticEnrichedPPLAveDissimilarity',
        'SemanticEnrichedMTEAveDissimilarity',
    ]

    methods = { 
        'general_baselines': [
        ],
        '$\\text{CoCoA}_{MSP}$': [
            'SemanticEnrichedMaxprobAveDissimilarity',
        ],
        '$\\text{CoCoA}_{PPL}$': [
            'SemanticEnrichedPPLAveDissimilarity',
        ],
        '$\\text{CoCoA}_{MTE}$': [
            'SemanticEnrichedMTEAveDissimilarity',
        ]
    }

    if args.exclude_ss:
        for key, value in methods.items():
            changed_methods = []
            for method in value:
                if method not in single_sequence_methods:
                    changed_methods.append(method)
            methods[key] = changed_methods

    if args.do_sample:
        for key, value in methods.items():
            changed_methods = []
            for method in value:
                if method in single_sequence_methods:
                    changed_method = f'Sampled{method}'
                else:
                    changed_method = method

                if method in single_sequence_methods + focused_sample_methods:
                    if args.sample_strategy == 'first':
                        changed_method = changed_method
                    elif args.sample_strategy == 'best':
                        changed_method = f'Best{changed_method}'
                    elif args.sample_strategy == 'best_normalized':
                        changed_method = f'BestNormalized{changed_method}'
                    else:
                        raise ValueError(f'Invalid sample strategy: {args.sample_strategy}')

                changed_methods.append(changed_method)

            methods[key] = changed_methods
    else:
        for key, value in methods.items():
            changed_methods = []
            for method in value:
                if method in focused_sample_methods:
                    changed_method = f'Greedy{method}'
                else:
                    changed_method = method
                changed_methods.append(changed_method)
            methods[key] = changed_methods

    return methods

def parse_args():
    parser = argparse.ArgumentParser()
    # boolean argument do_sample with default value of False
    parser.add_argument('--do_sample', action='store_true', default=False)
    parser.add_argument('--exclude_ss', action='store_true')
    parser.add_argument('--sample_strategy', default='first')
    return parser.parse_args()


def build_rejection_curve(ues, metrics):
    order = np.argsort(ues)
    sorted_metrics = metrics[order]
    sum_rej_metrics = np.cumsum(sorted_metrics)
    num_points_left = np.arange(1, len(sum_rej_metrics) + 1)

    rej_metrics = sum_rej_metrics / num_points_left
    rej_rates = 1 - num_points_left / len(sum_rej_metrics)

    return rej_metrics[::-1], rej_rates[::-1]

def plot_rejection_curve(metrics, methods, man):
    model = 'mistral7b'
    metric = 'AlignScoreInputOutput'
    dataset = 'XSum'

    oracle_rejection, rates = build_rejection_curve(-metrics, metrics)
    plt.plot(rates, oracle_rejection, label='Oracle')

    for method in methods:
        ues = man.estimations[('sequence', method)]
        ue_rejection, rates = build_rejection_curve(ues, metrics)
        plt.plot(rates, ue_rejection, label=method)
    plt.legend()
    plt.xlabel('Rejection Rate')
    plt.ylabel(metric)
    plt.title(f'{model} {dataset} {metric}')
    plt.savefig(f'xsum_rejection.png')
    plt.close()

def hex_to_rgb(hex):
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def cell_format(prop):
    _, hex = prop[0].split('#')
    rgb = list(hex_to_rgb(hex))

    a = 0.5
    R,G,B = 255, 255, 255

    rgb[0] = rgb[0] * a + (1.0 - a) * R
    rgb[1] = rgb[1] * a + (1.0 - a) * G
    rgb[2] = rgb[2] * a + (1.0 - a) * B

    return '\\cellcolor[rgb]{' + ', '.join([str(x/255) for x in rgb]) + '}'

def text_format(prop):
    _, hex_with_content = prop[0].split('#')
    hex, content, _ = hex_with_content.split(' ')
    rgb = hex_to_rgb(hex)

    #return '\\textcolor[rgb]{' + ', '.join([str(x/255) for x in rgb]) + '}{' + content.strip() + '} '
    return content.strip()

def postprocess_latex(latex, metric_row):
    #latex = latex.replace('_', '\_')
    latex = latex.replace('table', 'table*')
    latex = re.sub(r'\\background-color#[A-Za-z0-9]{6}', cell_format, latex)
    latex = re.sub(r'\\color#[A-Za-z0-9]{6} -?\d\.\d* ', text_format, latex)
    latex = latex.splitlines()

    split_methods = [
        'MaximumSequenceProbability &',
        'TokenSAR &',
        'Perplexity &',
        'MeanTokenEntropy &',
        'SampledMaximumSequenceProbability &',
        'SampledTokenSAR &',
        'SampledPerplexity &',
        'SampledMeanTokenEntropy &',
        'BestSampledMaximumSequenceProbability &',
        'BestSampledTokenSAR &',
        'BestSampledPerplexity &',
        'BestSampledMeanTokenEntropy &',
        'BestNormalizedSampledMaximumSequenceProbability &',
        'BestNormalizedSampledTokenSAR &',
        'BestNormalizedSampledPerplexity &',
        'BestNormalizedSampledMeanTokenEntropy &',
    ]

    new_latex = []
    for i, line in enumerate(latex):
        if i == 0:
            new_latex.append(line)
            new_latex.append('\\footnotesize')
            new_latex.append('\\centering')
        #elif line.startswith('\\begin{tabular}'):
            #new_line = ' & '.join([str(round(m, 3)) for m in metric_row])
            #new_line = f'Mean metric & {new_line} \\\\'
            #new_latex.append(line)
            #new_latex.append(new_line)
        #elif any([line.startswith(m) for m in split_methods]):
        #    new_latex.append('\\midrule')
        #    new_latex.append(line)
        else:
            new_latex.append(line)

    latex = '\n'.join(new_latex)

    return latex

def strip_latex(latex, label):
    latex = latex.splitlines()
    header = latex[0:6]
    footer = [latex[-2]] + [label] + [latex[-1]]
    group_rows = latex[6:-2]

    return header, group_rows, footer

def get_caption(model, args):
    if args.do_sample:
        if args.sample_strategy == 'first':
            prefix = 'First Sample'
        elif args.sample_strategy == 'best':
            prefix = 'Best Sample'
        elif args.sample_strategy == 'best_normalized':
            prefix = 'Best Normalized Sample'
        else:
            raise ValueError(f'Invalid sample strategy: {args.sample_strategy}')
    else:
        prefix = 'Greedy'

    return f'{prefix} PRRs for {model}, comparison between CoCoA family methods based on different choices of similarity function'

def get_label(model, args):
    prefix = args.sample_strategy if args.do_sample else 'greedy'
    label =  f'{model}_{prefix}_sim_mat_ablation'
    return '\\label{' + label + '}'

class SkeletonManager:
    def __init__(self, metrics):
        self.metrics = metrics

def main():
    args = parse_args()
    base_dir = f'sample_metric_mans/with_extra_sim_matrices_enriched'

    mans = defaultdict(lambda: defaultdict(dict))

    for model in models:
        for dataset in datasets:
            for sim_mat_type in sim_mat_types:
                metrics = UEManager.load(f'{base_dir}/{sim_mat_type}/{model}_{dataset}.man').metrics
                mans[model][dataset][sim_mat_type] = SkeletonManager(metrics)


    for do_sample in [True, False]:
        args.do_sample = do_sample
        if args.do_sample:
            args.sample_strategy = 'best'

        args.ablation_type = 'sim_mat_ablation'
        
        model_blocks = []
        for model in models:
            out_dir = Path(f'{base_dir}') / f'ablation/{args.ablation_type}'
            out_dir.mkdir(parents=True, exist_ok=True)

            if args.do_sample:
                if args.sample_strategy == 'best':
                    tex_prefix = 'best_sample'
                else:
                    raise ValueError(f'Invalid sample strategy: {args.sample_strategy}')
            else:
                tex_prefix = 'greedy'

            methods_dict = get_methods(args)
            ats_metrics, nmt_metrics, short_qa_metrics, long_qa_metrics = get_metrics(args)

            metric_row = []

            all_rows = []
            for i, (group_name, methods) in enumerate(methods_dict.items()):
                group_rows = {}
                for method in methods:
                    for sim_mat_type in sim_mat_types:
                        method_row = []
                        for metric in ats_metrics:
                            man = mans[model]['xsum'][sim_mat_type]
                            prr = man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                            method_row.append(prr)
                        for metric in nmt_metrics:
                            man = mans[model]['wmt14_fren'][sim_mat_type]
                            prr = man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                            method_row.append(prr)

                            man = mans[model]['wmt19_deen'][sim_mat_type]
                            prr = man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                            method_row.append(prr)
                        for metric in long_qa_metrics:
                            man = mans[model]['coqa'][sim_mat_type]
                            prr = man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                            method_row.append(prr)

                            man = mans[model]['trivia'][sim_mat_type]
                            prr = man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                            method_row.append(prr)
                        for metric in short_qa_metrics:
                            man = mans[model]['mmlu'][sim_mat_type]
                            prr = man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                            method_row.append(prr)

                            man = mans[model]['gsm8k_cot'][sim_mat_type]
                            prr = man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                            method_row.append(prr)

                        row_name = method_names[method] if method in method_names else method
                        #row_name = f'{row_name} ({sim_mat_type})'
                        row_name = sim_mat_names[sim_mat_type]
                        group_rows[row_name] = method_row

                df = pd.DataFrame.from_dict(group_rows, orient='index', columns=('XSum', 'WMT14FrEn', 'WMT19DeEn', 'CoQa', 'Trivia', 'MMLU', 'GSM8k',))
                caption = get_caption(model, args)
                latex = df.style.format(precision=3).background_gradient(cmap=cm).set_caption(caption).to_latex()
                latex = postprocess_latex(latex, metric_row)
                header, latex_group_rows, footer = strip_latex(latex, get_label(model, args))

                if len(latex_group_rows) > 0:
                    group_header = [
                        '\n\\midrule\n',
                        '& \\multicolumn{7}{c}{',
                        group_name,
                        '}\\\\\n\\midrule\n'
                    ]

                    all_rows = all_rows + group_header + latex_group_rows
                else:
                    continue

            model_row = '& \\multicolumn{7}{c}{' + model_names[model] + '} \\\\'
            model_blocks.append(model_row + '\n' + '\n'.join(all_rows) + '\n\\midrule\n')

        table = '\n'.join(header + ['\n\\midrule\n'] + model_blocks + footer)

        with open(f'{out_dir}/{tex_prefix}_ablation.tex', 'w') as f:
            f.write(table)

if __name__ == '__main__':
    main()
