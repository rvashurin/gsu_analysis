import argparse
import pathlib
import os
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.estimators import *
from lm_polygraph.ue_metrics import *
from lm_polygraph.stat_calculators import *
from tqdm import tqdm

from transformers import AutoTokenizer

class DummyNLI:
    def __init__(self):
        self.batch_size = 10

class DummyModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('mistral-community/Mistral-7B-v0.2')

    def device(self):
        return 'cpu'

calc = GreedySimilarityCalculator(DummyNLI(),)

in_dir = '/workspace/mans'
out_dir = '/workspace/processed_mans'

models = ["mistral7b", "llama8b"]
datasets = [
    "trivia", "mmlu", "coqa", "gsm8k_cot", 
    "xsum", "wmt14_fren", "wmt19_deen", 
    "wmt14_enfr", "wmt19_ende"
]

for model in models:
    for dataset in datasets:
        man = UEManager.load(f'{in_dir}/{model}_{dataset}.man')

        stats = man.stats
        texts = man.stats['greedy_texts']

        out = calc(dependencies=stats, texts=texts, model=DummyModel())

        man.stats.update(out)

        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        man.save(os.path.join(out_dir, f"{model}_{dataset}.man"))
