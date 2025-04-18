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
    parser.add_argument('--mt5_device', type=str, default='cuda:0')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--script_dir', type=str, default='/workspace/mans')
    parser.add_argument('--out_dir', type=str, default='/workspace/mans_enriched')

    return parser.parse_args()


def extract_raw_inputs(dataset, input_texts):
    raw_inputs = []

    if dataset == 'trivia':
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

    stat_calculators = [
        FirstSampleCalculator(),
        BestSampleCalculator(),
    ]

    model_name_or_path="google/metricx-24-hybrid-large-v2p6"
    tokenizer_name="google/mt5-large"
    device = torch.device(args.mt5_device)
    model_xmetric = MT5ForRegression.from_pretrained(model_name_or_path)
    model_xmetric.to(device)
    model_xmetric.eval()
    tokenizer_xmetric = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path
    )

    gen_metrics_wmt = [
        XMetric(model=model_xmetric,
                tokenizer=tokenizer_xmetric,
                source_ignore_regex="(?s).*Original:\n(.*?)\nTranslation:\n"),
        XMetric(model=model_xmetric,
                tokenizer=tokenizer_xmetric,
                source_ignore_regex="(?s).*Original:\n(.*?)\nTranslation:\n",
                sample=True),
        XMetric(model=model_xmetric,
                tokenizer=tokenizer_xmetric,
                source_ignore_regex="(?s).*Original:\n(.*?)\nTranslation:\n",
                sample=True,
                sample_strategy='Best'),
        XMetric(model=model_xmetric,
                tokenizer=tokenizer_xmetric,
                source_ignore_regex="(?s).*Original:\n(.*?)\nTranslation:\n",
                sample=True,
                sample_strategy='BestNormalized'),
    ]

    gen_metrics_qa = [
        GptAccuracyMetric(api_key=args.api_key),
        GptAccuracyMetric(api_key=args.api_key, sample=True),
        GptAccuracyMetric(api_key=args.api_key, sample=True, sample_strategy='Best'),
        GptAccuracyMetric(api_key=args.api_key, sample=True, sample_strategy='BestNormalized'),
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

            if 'wmt' in dataset:
                for gen_metric in gen_metrics_wmt:
                    values = gen_metric(stats=man.stats, target_texts=stats['target_texts'])
                    man.gen_metrics[('sequence', str(gen_metric))] = values
            elif dataset in ['coqa', 'gsm8k_cot', 'trivia', 'mmlu']:
                for gen_metric in gen_metrics_qa:
                    values = gen_metric(stats=man.stats, target_texts=stats['target_texts'])
                    man.gen_metrics[('sequence', str(gen_metric))] = values

            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            man.save_path = os.path.join(out_dir, f"{model}_{dataset}.man")
            man.save()

if __name__ == '__main__':
    args = parse_args()
    main(args)
