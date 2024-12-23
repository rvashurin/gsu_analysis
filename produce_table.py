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

models = ['mistral7b', 'llama8b']

def get_metrics(args):
    if args.do_sample:
        ats_metrics = ['SampleRouge_rougeL']
        nmt_metrics = ['SampleComet']
        short_qa_metrics = ['SampleAccuracy']
        long_qa_metrics = ['SampleAlignScoreOutputTarget']
    else:
        ats_metrics = ['Rouge_rougeL']
        nmt_metrics = ['Comet']
        short_qa_metrics = ['Accuracy']
        long_qa_metrics = ['AlignScoreOutputTarget']

    return ats_metrics, nmt_metrics, short_qa_metrics, long_qa_metrics

def get_methods(args):
    single_sequence_methods = [
        'MaximumSequenceProbability',
        'TokenSAR',
        'Perplexity',
        'MeanTokenEntropy',
    ]

    methods = { 
        'general_baselines': [
            'MonteCarloSequenceEntropy',
            'MonteCarloNormalizedSequenceEntropy',
            'SemanticEntropy',
            'CEDegMat',
        ],
        'msp': [
            'MaximumSequenceProbability',
            'AveMaxprob',
            'SentenceSAR',
            'MaxprobGSU',
            'MaxprobGSUexp',
            'SemanticAveMaxprob',
            'SemanticAveMaxprobexp',
        ],
        'tsar': [
            'TokenSAR',
            'AveTokenSAR',
            'SAR_t0.001',
            'TokenSARGSU',
            'TokenSARGSUexp',
            'SemanticAveTokenSAR',
            'SemanticAveTokenSARexp',
        ],
        'ppl': [
            'Perplexity',
            'AvePPL',
            'PPLSAR',
            'PPLGSU',
            'PPLGSUexp',
            'SemanticAvePPL',
            'SemanticAvePPLexp',
        ],
        'mte': [
            'MeanTokenEntropy',
            'AveMTE',
            #'MTESAR',
            'MTEGSU',
            'SemanticAveMTE',
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
        change_methods = single_sequence_methods

        for key, value in methods.items():
            changed_methods = []
            for method in value:
                if method in change_methods:
                    changed_methods.append(f'Sampled{method}')
                else:
                    changed_methods.append(method)
            methods[key] = changed_methods

    return methods

def parse_args():
    parser = argparse.ArgumentParser()
    # boolean argument do_sample with default value of False
    parser.add_argument('--do_sample', action='store_true', default=False)
    parser.add_argument('--exclude_ss', action='store_true')
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
    latex = latex.replace('_', '\_')
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
        'SampledMeanTokenEntropy &'
    ]

    new_latex = []
    for i, line in enumerate(latex):
        if i == 0:
            new_latex.append(line)
            new_latex.append('\\footnotesize')
        elif line.startswith('\\begin{tabular}'):
            new_line = ' & '.join([str(round(m, 3)) for m in metric_row])
            new_line = f'Mean metric & {new_line} \\\\'
            new_latex.append(line)
            new_latex.append(new_line)
        elif any([line.startswith(m) for m in split_methods]):
            new_latex.append('\\midrule')
            new_latex.append(line)
        else:
            new_latex.append(line)

    latex = '\n'.join(new_latex)

    return latex

def strip_latex(latex):
    latex = latex.splitlines()
    header = latex[0:4] + latex[5:6]
    footer = latex[-2:]
    metric_row = latex[4]
    group_rows = latex[6:-2]

    return header, metric_row, group_rows, footer

def main():
    args = parse_args()

    for model in models:
        #if args.do_sample:
        base_dir = 'sample_metric_mans/log_exp'
        #else:
        #    base_dir = 'greedy_metric_mans/log_exp'
        tex_prefix = 'sample' if args.do_sample else 'greedy'

        methods_dict = get_methods(args)
        ats_metrics, nmt_metrics, short_qa_metrics, long_qa_metrics = get_metrics(args)

        trivia_man = UEManager.load(f'{base_dir}/{model}_trivia.man')
        mmlu_man = UEManager.load(f'{base_dir}/{model}_mmlu.man')
        coqa_man = UEManager.load(f'{base_dir}/{model}_coqa.man')
        gsm8k_man = UEManager.load(f'{base_dir}/{model}_gsm8k_cot.man')

        xsum_man = UEManager.load(f'{base_dir}/{model}_xsum.man')

        wmt_14_fren_man = UEManager.load(f'{base_dir}/{model}_wmt14_fren.man')
        wmt_19_deen_man = UEManager.load(f'{base_dir}/{model}_wmt19_deen.man')

        # Translation tasks
        metric_row = []
        for metric in nmt_metrics:
            mean_metric = np.mean(wmt_14_fren_man.gen_metrics[('sequence', metric)])
            metric_row.append(mean_metric)
            mean_metric = np.mean(wmt_19_deen_man.gen_metrics[('sequence', metric)])
            metric_row.append(mean_metric)

        all_rows = []
        for _, methods in methods_dict.items():
            group_rows = {}
            for method in methods:
                method_row = []
                for metric in nmt_metrics:
                    prr = wmt_14_fren_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                    method_row.append(prr)
                    prr = wmt_19_deen_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                    method_row.append(prr)
                group_rows[method] = method_row

            df = pd.DataFrame.from_dict(group_rows, orient='index', columns=('WMT14FrEn', 'WMT19DeEn'))
            latex = df.style.format(precision=3).background_gradient(cmap=cm).set_caption(f'{model} PRRs').to_latex()
            latex = postprocess_latex(latex, metric_row)
            header, latex_metric_row, latex_group_rows, footer = strip_latex(latex)
            all_rows = all_rows + latex_group_rows

        table = '\n'.join(header + [latex_metric_row] + all_rows + footer)

        with open(f'{base_dir}/{tex_prefix}_{model}_sentsar_nmt.tex', 'w') as f:
            f.write(table)

        # Summarization tasks
        metric_row = []
        for metric in ats_metrics:
            mean_metric = np.mean(xsum_man.gen_metrics[('sequence', metric)])
            metric_row.append(mean_metric)

        all_rows = []
        for _, methods in methods_dict.items():
            group_rows = {}
            for method in methods:
                method_row = []
                for metric in ats_metrics:
                    prr = xsum_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                    method_row.append(prr)
                group_rows[method] = method_row

            df = pd.DataFrame.from_dict(group_rows, orient='index', columns=('XSum/RougeL',))
            latex = df.style.format(precision=3).background_gradient(cmap=cm).set_caption(f'{model} PRRs').to_latex()
            latex = postprocess_latex(latex, metric_row)
            header, latex_metric_row, latex_group_rows, footer = strip_latex(latex)
            all_rows = all_rows + latex_group_rows

        table = '\n'.join(header + [latex_metric_row] + all_rows + footer)
        with open(f'{base_dir}/{tex_prefix}_{model}_sentsar_ats.tex', 'w') as f:
            f.write(table)

        # QA tasks
        metric_row = []
        for metric in long_qa_metrics:
            mean_metric = np.mean(coqa_man.gen_metrics[('sequence', metric)])
            metric_row.append(mean_metric)
            mean_metric = np.mean(trivia_man.gen_metrics[('sequence', metric)])
            metric_row.append(mean_metric)
        for metric in short_qa_metrics:
            mean_metric = np.mean(mmlu_man.gen_metrics[('sequence', metric)])
            metric_row.append(mean_metric)
            mean_metric = np.mean(gsm8k_man.gen_metrics[('sequence', metric)])
            metric_row.append(mean_metric)

        all_rows = []
        for _, methods in methods_dict.items():
            group_rows = {}
            for method in methods:
                method_row = []
                for metric in long_qa_metrics:
                    prr = coqa_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                    method_row.append(prr)
                    prr = trivia_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                    method_row.append(prr)
                for metric in short_qa_metrics:
                    prr = mmlu_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                    method_row.append(prr)
                    prr = gsm8k_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                    method_row.append(prr)
                group_rows[method] = method_row

            df = pd.DataFrame.from_dict(group_rows, orient='index', columns=('CoQa/Al', 'Trivia/Al', 'MMLU/Acc', 'GSM8k/Acc',))
            latex = df.style.format(precision=3).background_gradient(cmap=cm).set_caption(f'{model} PRRs').to_latex()
            latex = postprocess_latex(latex, metric_row)
            header, latex_metric_row, latex_group_rows, footer = strip_latex(latex)
            all_rows = all_rows + latex_group_rows

        table = '\n'.join(header + [latex_metric_row] + all_rows + footer)

        with open(f'{base_dir}/{tex_prefix}_{model}_sentsar_qa.tex', 'w') as f:
            f.write(table)

if __name__ == '__main__':
    main()