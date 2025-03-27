import argparse
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.estimators import *
from lm_polygraph.ue_metrics import *
from lm_polygraph.stat_calculators import *
from lm_polygraph.generation_metrics import *
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.generation_metrics.x_metric_utils import MT5ForRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


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


def main():
    # Define models and datasets
    models = ["falcon7b", "mistral7b", "llama8b"]
    #models = ["falcon7b"]
    datasets = [
        "trivia", "mmlu", "coqa", "gsm8k_cot", 
        "xsum", "wmt14_fren", "wmt19_deen", 
    ]

    script_dir = '/workspace/mans'
    out_dir = '/workspace/mans_enriched'

    nli_model = Deberta(batch_size=5, device='cuda:0')

    stat_calculators = [
        FirstSampleCalculator(),
        BestSampleCalculator(),
        SemanticMatrixCalculator(nli_model=nli_model),
        GreedySemanticMatrixCalculator(nli_model=nli_model),
        ConcatSemanticMatrixCalculator(nli_model=nli_model),
        ConcatGreedySemanticMatrixCalculator(nli_model=nli_model),
    ]

    model_name_or_path="google/metricx-24-hybrid-large-v2p6"
    tokenizer_name="google/mt5-large"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_xmetric = MT5ForRegression.from_pretrained(model_name_or_path)
    model_xmetric.to(device)
    model_xmetric.eval()
    tokenizer_xmetric = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path
    )

    gen_metrics = [
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

            for gen_metric in gen_metrics:
                values = gen_metric(stats=man.stats, target_texts=None)
                man.gen_metrics[('sequence', str(gen_metric))] = values

            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            man.save_path = os.path.join(out_dir, f"{model}_{dataset}.man")
            man.save()

if __name__ == '__main__':
    main()
