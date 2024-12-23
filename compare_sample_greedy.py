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

models = ['mistral7b', 'llama8b']

def main():
    for model in models:
        base_dir = 'sample_metric_mans'
        sample_mans = [
            UEManager.load(f'{base_dir}/{model}_trivia.man'),
            UEManager.load(f'{base_dir}/{model}_mmlu.man'),
            UEManager.load(f'{base_dir}/{model}_coqa.man'),
            UEManager.load(f'{base_dir}/{model}_gsm8k_cot.man'),
        ]

        base_dir = 'greedy_metric_mans/mte_fixed_ave_appended'
        greedy_mans = [
            UEManager.load(f'{base_dir}/{model}_trivia.man'),
            UEManager.load(f'{base_dir}/{model}_mmlu.man'),
            UEManager.load(f'{base_dir}/{model}_coqa.man'),
            UEManager.load(f'{base_dir}/{model}_gsm8k_cot.man'),
        ]

        for i, (sample_man, greedy_man) in enumerate(zip(sample_mans, greedy_mans)):
            sample_metrics = np.array(sample_man.gen_metrics[('sequence', 'SampleAccuracy')])
            greedy_metrics = np.array(greedy_man.gen_metrics[('sequence', 'Accuracy')])

            targets = np.array(sample_man.stats['target_texts'], dtype=object)
            sample_out = np.array([s[0] for s in sample_man.stats['sample_texts']])
            greedy_out = np.array(greedy_man.stats['greedy_texts'])

            greedy_right = np.argwhere(greedy_metrics > sample_metrics)
            greedy_right_greedy_preds = greedy_out[greedy_right]
            greedy_right_sample_preds = sample_out[greedy_right]
            greedy_right_targets = targets[greedy_right]
            greedy_right_tuples = list(zip(greedy_right_targets, greedy_right_sample_preds, greedy_right_greedy_preds))

            sample_right = np.argwhere(greedy_metrics < sample_metrics)
            sample_right_greedy_preds = greedy_out[sample_right]
            sample_right_sample_preds = sample_out[sample_right]
            sample_right_targets = targets[sample_right]
            sample_right_tuples = list(zip(sample_right_targets, sample_right_sample_preds, sample_right_greedy_preds))


if __name__ == '__main__':
    main()
