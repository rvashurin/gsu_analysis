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
        self.batch_size = 1

class DummyModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('mistral-community/Mistral-7B-v0.2')

    def device(self):
        return 'cpu'

calc = GreedySimilarityCalculator(DummyNLI(),)

out_dir = 'sample_metric_mans/best_sample_enriched'

man = UEManager.load(f'{out_dir}/mistral7b_trivia.man')

stats = man.stats
texts = man.stats['greedy_texts']

calc(dependencies=stats, texts=texts, model=DummyModel())
