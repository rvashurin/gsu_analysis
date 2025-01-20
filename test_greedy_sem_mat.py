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

nli_model = Deberta(batch_size=1, device='cpu')

class DummyModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('mistral-community/Mistral-7B-v0.2')

    def device(self):
        return 'cpu'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
ckpt_path="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt"
align_scorer = AlignScorer(
    model="roberta-large",
    batch_size=batch_size,
    device=device,
    ckpt_path=ckpt_path,
    evaluation_mode="nli_sp",
)

def main():
    # Define models and datasets
    models = ["falcon7b", "mistral7b", "llama8b"]
    datasets = [
        "trivia", "mmlu", "coqa", "gsm8k_cot", 
        "xsum", "wmt14_fren", "wmt19_deen", 
        #"wmt14_enfr", "wmt19_ende"
    ]
    #models = ["falcon7b"]
    #datasets = ["trivia"]

    script_dir = 'sample_metric_mans/best_sample_with_greedy'

    stat_calculators = [
        GreedyAlignMatrixCalculator(align_scorer),
    ]

    # Loop through each model and dataset combination
    for model in tqdm(models):
        for dataset in tqdm(datasets):
            # Construct manager file path
            manager_filename = f"{model}_{dataset}.man"
            manager_path = os.path.join(script_dir, manager_filename)

            man = UEManager.load(manager_path)

            stats = man.stats

            for calculator in stat_calculators:
                texts = stats["greedy_texts"]
                values = calculator(dependencies=stats, texts=texts, model=DummyModel())
                breakpoint()
                pass

if __name__ == '__main__':
    main()
