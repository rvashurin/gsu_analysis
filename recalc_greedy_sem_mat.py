import argparse
import pathlib
import os
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.estimators import *
from lm_polygraph.ue_metrics import *
from lm_polygraph.stat_calculators import *
from transformers import AutoTokenizer
from tqdm import tqdm
import torch 
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.generation_metrics.alignscore_utils import AlignScorer
import numpy as np
import pathlib

class DummyModel:
    def __init__(self, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained('mistral-community/Mistral-7B-v0.2')
        self.device = device

    def device(self):
        return self.device

def parse_args():
    parser = argparse.ArgumentParser()
    # boolean argument do_sample with default value of False
    parser.add_argument('--model', default='mistral7b')
    # list argument
    parser.add_argument('--datasets', nargs='+', default=["trivia", "mmlu", "coqa", "gsm8k_cot", "xsum", "wmt14_fren", "wmt19_deen"])
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    # required arguments
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    model = args.model
    datasets = args.datasets
    in_dir = args.in_dir
    out_dir = args.out_dir
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    device = torch.device(f"cuda:{cuda_device}")

    ckpt_path="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt"
    align_scorer = AlignScorer(
        model="roberta-large",
        batch_size=batch_size,
        device=device,
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",
    )
    nli_model = Deberta(batch_size=batch_size, device=f"cuda:{cuda_device}")

    stat_calculators = [
        SemanticMatrixCalculator(nli_model),
        AlignMatrixCalculator(align_scorer),
        RougeLSemanticMatrixCalculator(),
#
        GreedySimilarityCalculator(nli_model),
        GreedySemanticMatrixCalculator(nli_model),
        GreedyAlignMatrixCalculator(align_scorer),
        GreedyRougeLSemanticMatrixCalculator(),
    ]

    # Loop through datasets
    for dataset in tqdm(datasets):
        # Construct manager file path
        manager_filename = f"{model}_{dataset}.man"
        manager_path = os.path.join(in_dir, manager_filename)

        man = UEManager.load(manager_path)

        stats = man.stats

        for calculator in stat_calculators:
            texts = stats["greedy_texts"]
            values = calculator(dependencies=stats, texts=texts, model=DummyModel(device=f"cuda:{cuda_device}",))
            stats.update(values)

        man.stats = stats
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        man.save_path = os.path.join(out_dir, f"{model}_{dataset}.man")
        man.save()

if __name__ == '__main__':
    main()
