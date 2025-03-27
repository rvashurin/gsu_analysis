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

models = ['llama8b','mistral7b', 'falcon7b']

def get_metrics(args):
    if args.do_sample:
        if args.sample_strategy == 'first':
            ats_metrics = ['SampleRouge_rougeL']
            nmt_metrics = ['SampleComet']
            short_qa_metrics = ['SampleAccuracy']
            long_qa_metrics = ['SampleAlignScoreOutputTarget']
        elif args.sample_strategy == 'best':
            ats_metrics = ['BestSampleAlignScoreInputOutput']
            nmt_metrics = ['BestSampleComet']
            short_qa_metrics = ['BestSampleAccuracy']
            long_qa_metrics = ['BestSampleAlignScoreOutputTarget']
        elif args.sample_strategy == 'best_normalized':
            ats_metrics = ['BestNormalizedSampleRouge_rougeL']
            nmt_metrics = ['BestNormalizedSampleComet']
            short_qa_metrics = ['BestNormalizedSampleAccuracy']
            long_qa_metrics = ['BestNormalizedSampleAlignScoreOutputTarget']
        else:
            raise ValueError(f'Invalid sample strategy: {args.sample_strategy}')
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
            'SAR_t0.001'
        ],
        'msp': [
            'MaximumSequenceProbability',
            'GreedySemanticEnrichedMaxprobAveDissimilarity',

        ],
        'ppl': [
            'Perplexity',
            'GreedySemanticEnrichedPPLAveDissimilarity',

        ],
        'mte': [
            'MeanTokenEntropy',
            'GreedySemanticEnrichedMTEAveDissimilarity',
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



def get_tasks():

    tasks = { 
        'qa': [
            'trivia',
            'mmlu',
            'coqa',
            'gsm8k',
        ],
        'ats': [
            'wmt_14_fren',
            'wmt_19_deen',
        ],
        'sum': [
            'xsum',
        ]
    }
    return tasks    

def parse_args():
    parser = argparse.ArgumentParser()
    # boolean argument do_sample with default value of False
    parser.add_argument('--do_sample', action='store_true', default=False)
    parser.add_argument('--exclude_ss', action='store_true')
    parser.add_argument('--sample_strategy', default='first')
    return parser.parse_args()

def main():
    args = parse_args()

    # dict_results = {}
    results ={}
    for model in models:
        #if args.do_sample:
        base_dir = 'sample_metric_mans/final_mans'
        #else:
        #    base_dir = 'greedy_metric_mans/log_exp'
        tex_prefix = 'final_table' if args.do_sample else 'greedy'

        methods_dict = get_methods(args)
        tasks_dict = get_tasks()
        ats_metrics, nmt_metrics, short_qa_metrics, long_qa_metrics = get_metrics(args)

        trivia_man = UEManager.load(f'{base_dir}/{model}_trivia.man')
        mmlu_man = UEManager.load(f'{base_dir}/{model}_mmlu.man')
        coqa_man = UEManager.load(f'{base_dir}/{model}_coqa.man')
        gsm8k_man = UEManager.load(f'{base_dir}/{model}_gsm8k_cot.man')

        xsum_man = UEManager.load(f'{base_dir}/{model}_xsum.man')
        # xsum_man = None
        wmt_14_fren_man = UEManager.load(f'{base_dir}/{model}_wmt14_fren.man')
        wmt_19_deen_man = UEManager.load(f'{base_dir}/{model}_wmt19_deen.man')
        
        for _, methods in methods_dict.items():
            group_rows = {}
            # Translation
            for method in methods:
                method_row = []
                if method not in results:
                    results[method] = {}
                if model not in results[method]:
                    results[method][model] = {}

                for metric in nmt_metrics:
                    # print(f"{method}, {metric}: " ,wmt_14_fren_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')])
                    prr = wmt_14_fren_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                    method_row.append(prr)
                    # print(f"{method}, {metric}: " , wmt_19_deen_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')])
                    prr = wmt_19_deen_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                    method_row.append(prr)
                results[method][model]["nmt"] = np.mean(method_row)
                # print(f"{model}, {method}, nmt: {np.mean(task_performance)}")
            
            # Summ
            for _, methods in methods_dict.items():
                group_rows = {}
                for method in methods:
                    method_row = []
                    if method not in results:
                        results[method] = {}
                    if model not in results[method]:
                        results[method][model] = {}

                    for metric in ats_metrics:
                        prr = xsum_man.metrics[('sequence', method, metric, 'prr_0.5_normalized')]
                        method_row.append(prr)
                    results[method][model]["sum"] =np.mean(method_row)
                    # print(f"{model}, {method}, sum: {np.mean(method_row)}")
            
            for _, methods in methods_dict.items():
                group_rows = {}
                for method in methods:
                    method_row = []
                    if method not in results:
                        results[method] = {}
                    if model not in results[method]:
                        results[method][model] = {}
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
                    results[method][model]["qa"] = np.mean(method_row)
                    # print(f"{model}, {method}, qa: {np.mean(method_row)}")
    
    # print(results)
    tasks = ['qa', 'nmt', 'sum']

    for model in models:
        for task in tasks:
            # Collect all methods' scores for this model and task
            task_scores = {}
            for category, method_list in methods_dict.items():
                for method in method_list:
                    task_scores[method] = results[method][model][task]

            # Find the best and second-best methods
            sorted_methods = sorted(task_scores.items(), key=lambda x: x[1], reverse=True)
            best_method, best_score = sorted_methods[0]
            second_best_method, second_best_score = sorted_methods[1] if len(sorted_methods) > 1 else (None, None)

            # Update results with formatted values
            for method, score in task_scores.items():
                if score == best_score:
                    results[method][model][task] = f"\\textbf{{{score:.3f}}}"
                elif score == second_best_score:
                    results[method][model][task] = f"\\underline{{{score:.3f}}}"
                else:
                    results[method][model][task] = f"{score:.3f}"

    header = """
    \\begin{table*}[th!]
    \\centering
    \\renewcommand{\\arraystretch}{1.2} % Adjust row height
    \scalebox{0.85}{
    \\begin{tabular}{lccccccccc}
    \\bottomrule
    \\textbf{Metric} & \multicolumn{3}{c}{\\textbf{Llama}} & \multicolumn{3}{c}{\\textbf{Mistral}} & \multicolumn{3}{c}{\\textbf{Falcon}} \\\\  
    \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
    & \\textbf{QA} & \\textbf{NMT} & \\textbf{SUM} 
    & \\textbf{QA} & \\textbf{NMT} & \\textbf{SUM}  
    & \\textbf{QA} & \\textbf{NMT} & \\textbf{SUM}  \\\\
    \midrule
    """
    end_txt="""
    \\bottomrule
    \\end{tabular}}
    \\caption{Results for Evaluated Sequence - Greedy Sample: Mean PRR across datasets for each task. The best performing method is in bold, and the second-best is underscored. Arrows indicate improvement in CoCoA over the base version.}
    \\label{tab:best_sample_results}
    \\end{table*}"""


    method_mapping ={
    'MonteCarloSequenceEntropy': 'MCSE',
    'MonteCarloNormalizedSequenceEntropy': 'MCNSE',
    'SemanticEntropy': 'Semantic Entropy',
    'CEDegMat': 'CEDegMat',
    'SAR_t0.001': 'SAR',
    'MaximumSequenceProbability': 'MSP',
    'GreedySemanticEnrichedMaxprobAveDissimilarity': 'MSP',
    'Perplexity': 'Perplexity',
    'GreedySemanticEnrichedPPLAveDissimilarity': 'Perplexity',
    'MeanTokenEntropy': 'MeanTokenEntropy',
    'GreedySemanticEnrichedMTEAveDissimilarity': 'MeanTokenEntropy',
    }

    rows = []
    for _, methods in methods_dict.items():
        for method in methods:
            # Replace the short method name with the full method name
            full_method_name = method_mapping[method]  # Default to the original if no mapping is found
            if "Enriched" in method:
                row = "$\\text{" + f"{full_method_name}" + "}_{CoCoA}$" 
            else:
                row = "$\\text{" + f"{full_method_name}" + "}$"
            for model in models:
                for task in tasks:
                    if "Enriched" in method:
                        row = row + " & " + results[method][model][task] + "  \\(\\uparrow\\)  " 
                    else:
                        row = row + " & " + results[method][model][task]
            row = row + " \\\\"  # Add the newline for LaTeX table formatting
            if method == methods[-1] and _!='mte':
                row = row + " \\midrule"
            rows.append(row)
    
    combined_text = header + "\n".join(rows) + end_txt
    output_file = "greedy_output.txt"
    with open(output_file, "w") as file:
        file.write(combined_text)


    
if __name__ == '__main__':
    main()
