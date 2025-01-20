import numpy as np
from matplotlib.colors import ListedColormap
from lm_polygraph.utils.manager import UEManager
import tabulate
import pandas as pd
from functools import partial
import seaborn as sns
import re
import matplotlib.pyplot as plt

cm = sns.color_palette("coolwarm", as_cmap=True)

models = ['mistral']
gen_metrics = ('AlignScore')


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

#trivia_man = UEManager.load(f'mistral7b_trivia_sentsar_variants.man')
#gsm8k_man = UEManager.load(f'mistral7b_gsm8k_cot.man')
#xsum_man = UEManager.load(f'mistral7b_xsum.man')
#
#wmt_14_fren_man = UEManager.load(f'mistral7b_wmt14_fren.man')
#wmt_14_enfr_man = UEManager.load(f'mistral7b_wmt14_enfr.man')
#wmt_19_deen_man = UEManager.load(f'mistral7b_wmt19_deen.man')
#wmt_19_ende_man = UEManager.load(f'mistral7b_wmt19_ende.man')
#
#methods = ['MonteCarloSequenceEntropy',
#    'MonteCarloNormalizedSequenceEntropy',
#    'SemanticEntropy',
#    'MaximumSequenceProbability',
#    'SentenceSAR',
#    'MaxprobGSU_no_log_reverse',
#    'TokenSAR',
#    'SAR_t0.001',
#    'TokenSARGSU_no_log_reverse',
#    'Perplexity',
#    'PPLSAR',
#    'PPLGSU_no_log_reverse',
#    'MeanTokenEntropy',
#    'MTESAR',
#    'MTEGSU_no_log_reverse'
#]
#
#rows = {}
#
#managers = {
#    'wmt14_fren': wmt_14_fren_man,
#    'wmt19_deen': wmt_19_deen_man,
#    'gsm8k': gsm8k_man,
#    'xsum': xsum_man
#}

#for dataset_name, manager in managers.items():
manager = UEManager.load('sample_metric_mans/best_sample_with_greedy_enriched/mistral7b_wmt14_fren.man')

ppl_log_vals = np.array(manager.estimations[('sequence', 'Perplexity')])
ppl_exp_vals = -np.exp(-ppl_log_vals)

#log_vals = manager.estimations[('sequence', 'GreedySemanticEnrichedPPLAveDissimilarity')]
#exp_vals = manager.estimations[('sequence', 'GreedySemanticEnrichedPPLAveDissimilarityexp')]

#random_sims = np.random.uniform(0, 1, len(ppl_log_vals))
#
log_vals = ppl_log_vals * 0.05
exp_vals = ppl_exp_vals * 0.05

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

sns.histplot(ppl_log_vals, bins=100, ax=axs[0, 0], color='blue')
axs[0, 0].set_title('ppl log')

sns.histplot(ppl_exp_vals, bins=100, ax=axs[0, 1], color='red')
axs[0, 1].set_title('ppl exp')

# build histograms for MTE, MTE_SAR, MTE_GSU
sns.histplot(log_vals, bins=100, ax=axs[1, 0], color='blue')
axs[1, 0].set_title('mg_ppl_log')

sns.histplot(exp_vals, bins=100, ax=axs[1, 1], color='red')
axs[1, 1].set_title('mg_ppl_exp')

plt.suptitle(f'WMT14 FREN log/exp histograms')

plt.savefig(f'wmt14_fren_log_exp_hist.png')
