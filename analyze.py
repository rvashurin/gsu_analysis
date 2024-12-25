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
import pathlib
import os
from tqdm import tqdm
from scipy.stats import percentileofscore
from itertools import combinations

# set seeds
np.random.seed(1)

models = ['mistral7b', 'llama8b']

datasets = [
    "trivia", "mmlu", "coqa", "gsm8k_cot", 
    "xsum", "wmt14_fren", "wmt19_deen", 
]

script_dir = 'sample_metric_mans/log_exp'

heatmap_methods = [
    'CEDegMat',
    'MaximumSequenceProbability',
    'Perplexity',
    'MeanTokenEntropy',
    'SampledMaximumSequenceProbability',
    'SampledPerplexity',
    'SampledMeanTokenEntropy',
    'MonteCarloSequenceEntropy',
    'MonteCarloNormalizedSequenceEntropy',
    'SemanticEntropy',
    'AveMaxprob',
    'AvePPL',
    'AveMTE',
    'SemanticAveMaxprob',
    'SemanticAvePPL',
    'SemanticAveMTE',
    'SemanticAveMaxprobexp',
    'SemanticAvePPLexp',
    'SentenceSAR',
    'SAR_t0.001',
    'MaxprobGSU',
    'PPLGSU',
    'MTEGSU',
    'MaxprobGSUexp',
    'PPLGSUexp',
]

metrics = {
    'trivia': ['AlignScoreOutputTarget', 'SampleAlignScoreOutputTarget'],
    'coqa': ['AlignScoreOutputTarget', 'SampleAlignScoreOutputTarget'],
    'mmlu': ['Accuracy', 'SampleAccuracy'],
    'gsm8k_cot': ['Accuracy', 'SampleAccuracy'],
    'xsum': ['Rouge_rougeL', 'SampleRouge_rougeL'],
    'wmt14_fren': ['Comet', 'SampleComet'],
    'wmt19_deen': ['Comet', 'SampleComet'],
}

def plot_similarity_heatmap_with_legend(texts, similarity_matrix, figname, title):
    """
    Plots a heatmap of pairwise similarities with a numbered legend for texts.

    Parameters:
    - texts: List of text strings.
    - similarity_matrix: 2D numpy array representing pairwise similarities.
    """
    if len(texts) != similarity_matrix.shape[0]:
        raise ValueError("Number of texts must match the size of the similarity matrix.")

    # Create a heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(10, 9))  # Adjust figure size for better display

    #corr_matrix = pd.DataFrame(similarity_matrix, index=texts, columns=texts)
    corr_matrix = similarity_matrix

    # Plot the heatmap
    heatmap = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='none')

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman Correlation')
    
    ticks = range(len(texts))
    # Set ticks and labels
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(ticks, rotation=45, ha='left', fontsize=11)
    ax.set_yticklabels(ticks, fontsize=11)

    # Move x-ticks to the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Align ticks with cells
    ax.set_xticks(np.arange(-0.5, len(texts), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(texts), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Title and layout
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def plot_spearman_heatmap(data_dict, figname, title="Pairwise Spearman Rank Correlation Heatmap"):
    # Convert dictionary to a DataFrame
    df = pd.DataFrame(data_dict)

    # Compute the pairwise Spearman correlation matrix
    corr_matrix = df.corr(method='spearman')

    # Create a heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(18, 16))  # Adjust figure size for better display

    # Plot the heatmap
    heatmap = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='none')

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman Correlation')

    # Set ticks and labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='left', fontsize=11)
    ax.set_yticklabels(corr_matrix.columns, fontsize=11)

    # Move x-ticks to the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Align ticks with cells
    ax.set_xticks(np.arange(-0.5, len(corr_matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(corr_matrix.columns), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Title and layout
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


# Loop through each model and dataset combination
for model in tqdm(models):
    for dataset in tqdm(datasets):
        manager_filename = f"{model}_{dataset}.man"
        manager_path = os.path.join(script_dir, manager_filename)

        man = UEManager.load(manager_path)

        out_dir = pathlib.Path(f'eda/{model}_{dataset}')
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        fig_path = out_dir / f"method_corr_heatmap.png"
        data_dict = {k: man.estimations[('sequence', k)] for k in heatmap_methods}
        plot_spearman_heatmap(data_dict, fig_path, title=f"{model}_{dataset} Pairwise Spearman Rank Correlation Heatmap")

        data_size = len(man.stats['greedy_texts'])

        method_combinations = list(combinations(heatmap_methods, 2))
        for method, method2 in tqdm(method_combinations):
            pair_out_dir = out_dir / f"{method}_{method2}"

            if method == method2:
                continue
            # take 5 random examples that fall into lower 0.2 quantile according to one method and higher 0.8 quantile according to another method
            method_values = man.estimations[('sequence', method)]
            method2_values = man.estimations[('sequence', method2)]

            method_low_quantile = np.quantile(method_values, 0.3)
            method_high_quantile = np.quantile(method_values, 0.7)

            method2_low_quantile = np.quantile(method2_values, 0.3)
            method2_high_quantile = np.quantile(method2_values, 0.7)

            low_indices = np.where((method_values < method_low_quantile) & (method2_values > method2_high_quantile))[0]
            high_indices = np.where((method_values > method_high_quantile) & (method2_values < method2_low_quantile))[0]

            if len(low_indices) < 5 or len(high_indices) < 5:
                continue

            low_indices = np.random.choice(low_indices, 5, replace=False)
            high_indices = np.random.choice(high_indices, 5, replace=False)

            all_indices = np.concatenate([low_indices, high_indices])

            for i, random_i in enumerate(all_indices):
                # output to txt file both method values, quantiles
                pair_out_dir.mkdir(parents=True, exist_ok=True)
                method_value_quantile = percentileofscore(method_values, method_values[random_i])
                method2_value_quantile = percentileofscore(method2_values, method2_values[random_i])

                #matrix = np.array(man.stats['sample_sentence_similarity'][random_i])
                #samples = man.stats['sample_texts'][random_i]
                #fig_path = pair_out_dir / f"sample_similarity_{random_i}.png"
                #plot_similarity_heatmap_with_legend(samples, matrix, fig_path, title=f"{model}_{dataset} Sample {random_i} Pairwise Spearman Rank Correlation Heatmap")

                with open(pair_out_dir / f"{model}_{dataset}_sample_similarity_{method}_{method2}_{random_i}.txt", "w") as f:
                    text = ''
                    text += f"{model}_{dataset}\n"
                    text += f"Sample {random_i}\n\n"

                    text += f"=" * 100 + "\n\n"

                    text += f"{method} value: {method_values[random_i]:.3f}\n"
                    text += f"{method} value quantile: {method_value_quantile:.3f}\n\n"

                    text += f"{method2} value: {method2_values[random_i]:.3f}\n"
                    text += f"{method2} value quantile: {method2_value_quantile:.3f}\n\n"

                    text += f"=" * 100 + "\n\n"

                    text += "Metrics:\n"
                    for metric, values in man.gen_metrics.items():
                        if metric[1] in metrics[dataset]:
                            text += f"{metric[1]}: {values[random_i]:.3f}\n"

                    text += "\n" + f"=" * 100 + "\n\n"

                    text += f"target text: {man.stats['target_texts'][random_i]}\n\n"
                    text += f"greeedy text: {repr(man.stats['greedy_texts'][random_i])}\n\n"
                    text += f"sample text: {repr(man.stats['sample_texts'][random_i][0])}\n\n"

                    text += f"=" * 100 + "\n\n"

                    text += f"Samples:\n"

                    tab = []
                    for j, sample in enumerate(man.stats['sample_texts'][random_i]):
                        logprob = man.stats['sample_log_probs'][random_i][j]
                        norm_logprob = logprob / len(man.stats['sample_tokens'][random_i][j])
                        tab.append([j, repr(sample), logprob, norm_logprob])
                    text += tabulate.tabulate(tab, headers=["Index", "Sample", "Logprob", "Norm Logprob"], tablefmt="plain", floatfmt=".3f")

                    text += "\n\n" + f"=" * 100 + "\n\n"

                    text += f"Sentence Similarity Matrix:\n"
                    mat = list(man.stats['sample_sentence_similarity'][random_i])
                    for i, row in enumerate(mat):
                        mat[i] = [i] + list(row)
                    text += tabulate.tabulate(mat, headers=[""] + list(range(len(mat))), tablefmt="plain", floatfmt=".3f")

                    text += "\n\n" + f"=" * 100 + "\n\n"

                    text += f"Input:\n"
                    text += man.stats['input_texts'][random_i]

                    f.write(text)
