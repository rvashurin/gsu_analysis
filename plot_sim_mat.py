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

models = ['mistral7b', 'llama8b']

datasets = [
    "trivia", "mmlu", "coqa", "gsm8k_cot", 
    "xsum", "wmt14_fren", "wmt19_deen", 
]

script_dir = 'sample_metric_mans/log_exp'


def plot_similarity_heatmap_with_legend(texts, similarity_matrix, figname):
    """
    Plots a heatmap of pairwise similarities with a numbered legend for texts.

    Parameters:
    - texts: List of text strings.
    - similarity_matrix: 2D numpy array representing pairwise similarities.
    """
    if len(texts) != similarity_matrix.shape[0]:
        raise ValueError("Number of texts must match the size of the similarity matrix.")
    
    # Generate a numbered legend
    numbered_texts = [f"{i + 1}. {text}" for i, text in enumerate(texts)]
    
    # Create a figure with adjusted size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the heatmap
    cax = ax.matshow(similarity_matrix, cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Similarity Score')
    
    # Set axes labels
    ax.set_xticks(np.arange(len(texts)))
    ax.set_yticks(np.arange(len(texts)))
    ax.set_xticklabels(np.arange(1, len(texts) + 1))
    ax.set_yticklabels(np.arange(1, len(texts) + 1))
    
    # Add title and labels
    ax.set_title("Pairwise Similarity Heatmap", pad=20)
    ax.set_xlabel("Texts")
    ax.set_ylabel("Texts")
    
    # Add a legend as an inset
    props = dict(boxstyle="round", facecolor="white", edgecolor="black")
    legend_text = "\n".join(numbered_texts)
    fig.text(0.9, 0.5, legend_text, fontsize=10, va="center", ha="left", bbox=props)
    
    # Adjust layout to make room for the legend
    plt.subplots_adjust(right=0.8)  # Shrink the main plot to leave space for the legend
    plt.savefig(figname)

# Loop through each model and dataset combination
for model in tqdm(models):
    for dataset in tqdm(datasets):
        manager_filename = f"{model}_{dataset}.man"
        manager_path = os.path.join(script_dir, manager_filename)

        man = UEManager.load(manager_path)

        data_size = len(man.stats['greedy_texts'])
        random_i = np.random.randint(data_size)

        matrix = np.array(man.stats['sample_sentence_similarity'][random_i])
        samples = man.stats['sample_texts'][random_i]

        plot_similarity_heatmap_with_legend(samples, matrix, f"{model}_{dataset}_sample_similarity.png")
