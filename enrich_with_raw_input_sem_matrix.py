import argparse
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.estimators import *
from lm_polygraph.ue_metrics import *
from lm_polygraph.stat_calculators import *
from lm_polygraph.generation_metrics import *
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.generation_metrics.x_metric_utils import MT5ForRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='falcon7b')
    parser.add_argument('--datasets', nargs='+', default=['trivia', 'mmlu', 'coqa', 'gsm8k_cot', 'xsum', 'wmt14_fren', 'wmt19_deen'])
    parser.add_argument('--deberta_device', type=str, default='cuda:0')
    parser.add_argument('--script_dir', type=str, default='/workspace/mans')
    parser.add_argument('--out_dir', type=str, default='/workspace/mans_enriched')

    return parser.parse_args()


def extract_raw_inputs(dataset, input_texts):
    raw_inputs = []

    if dataset in ['trivia', 'coqa_no_context']:
        raw_inputs = ['\n'.join(text.split('\n')[-2:]) for text in input_texts]
    elif dataset == 'mmlu':
        raw_inputs = ['\n'.join(text.split('\n')[-6:]) for text in input_texts]
    elif dataset == 'coqa':
        raw_inputs = ['\n'.join([input_text.split('Question')[0]] + input_text.split('\n')[-2:]) for input_text in input_texts]
    elif dataset == 'gsm8k_cot':
        raw_inputs = ['\n'.join(text.split('\n')[-2:]) for text in input_texts]
    else:
        raw_inputs = input_texts

    return raw_inputs


def main(args):
    models = [args.model]
    datasets = args.datasets

    script_dir = args.script_dir
    out_dir = args.out_dir

    nli_model = Deberta(batch_size=1, device=args.deberta_device)

    stat_calculators = [
        FirstSampleCalculator(),
        BestSampleCalculator(),
        SemanticMatrixCalculator(nli_model=nli_model),
        GreedySemanticMatrixCalculator(nli_model=nli_model),
        ConcatSemanticMatrixCalculator(nli_model=nli_model),
        ConcatGreedySemanticMatrixCalculator(nli_model=nli_model),
    ]


    # Loop through each model and dataset combination
    for model in tqdm(models):
        for dataset in tqdm(datasets):
            # Construct manager file path
            manager_filename = f"{model}_{dataset}.man"
            manager_path = os.path.join(script_dir, manager_filename)

            man = UEManager.load(manager_path)

            stats = man.stats

            stats['no_fewshot_input_texts'] = extract_raw_inputs(dataset, stats['input_texts'])

            for key, value in stats.items():
                if isinstance(value, list) or isinstance(value, np.ndarray):
                    stats[key] = value[:5]

            for calculator in stat_calculators:
                texts = stats["greedy_texts"]
                values = calculator(dependencies=stats, texts=texts, model=None)
                stats.update(values)

            man.stats = stats

            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            man.save_path = os.path.join(out_dir, f"{model}_{dataset}.man")
            man.save()

if __name__ == '__main__':
    args = parse_args()
    main(args)
