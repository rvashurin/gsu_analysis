import argparse
import pathlib
import os
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.estimators import *
from lm_polygraph.ue_metrics import *
from lm_polygraph.stat_calculators import *
from tqdm import tqdm
from copy import deepcopy, copy

def main():
    # Define models and datasets
    models = ["falcon7b", "mistral7b", "llama8b"]
    datasets = [
        "trivia", "mmlu", "coqa", "gsm8k_cot", 
        "xsum", "wmt14_fren", "wmt19_deen", 
    ]

    sim_mat_types = ["align", "rouge", "nli", "ce"]

    sim_mat_keys = {
        "align": {
            "greedy": "greedy_align_semantic_matrix_forward",
            "sample": "align_semantic_matrix"
        },
        "rouge": {
            "greedy": "greedy_rouge_semantic_matrix",
            "sample": "rouge_semantic_matrix"
        },
        "nli": {
            "greedy": "greedy_semantic_matrix_forward",
            "sample": "semantic_matrix_entail"
        },
        "ce": {
            "greedy": "greedy_sentence_similarity_forward",
            "sample": "sample_sentence_similarity"
        }
    }

    base_keys = {
        "greedy": "greedy_sentence_similarity",
        "sample": "sample_sentence_similarity"
    }

    for sim_mat_type in sim_mat_types:
        script_dir = 'sample_metric_mans/with_extra_sim_matrices_enriched'
        out_dir = f'sample_metric_mans/with_extra_sim_matrices_enriched/{sim_mat_type}'

        estimators = [
            GreedySemanticEnrichedPPLAveDissimilarity(),
            GreedySemanticEnrichedMaxprobAveDissimilarity(),
            GreedySemanticEnrichedMTEAveDissimilarity(),
            SemanticEnrichedPPLAveDissimilarity(sample_strategy="best"),
            SemanticEnrichedMaxprobAveDissimilarity(sample_strategy="best"),
            SemanticEnrichedMTEAveDissimilarity(sample_strategy="best"),
        ]

        ue_metrics = [
            #PredictionRejectionArea(),
            PredictionRejectionArea(max_rejection=0.5),
        ]

        stat_calculators = [
            FirstSampleCalculator(),
            BestSampleCalculator(),
        ]

        # Loop through each model and dataset combination
        for model in tqdm(models):
            for dataset in tqdm(datasets):
                # Construct manager file path
                manager_filename = f"{model}_{dataset}.man"
                manager_path = os.path.join(script_dir, manager_filename)

                man = UEManager.load(manager_path)

                stats = man.stats
                old_estimations = deepcopy(man.estimations)

                stats[base_keys["greedy"]] = stats[sim_mat_keys[sim_mat_type]["greedy"]]
                stats[base_keys["sample"]] = stats[sim_mat_keys[sim_mat_type]["sample"]]

                for calculator in stat_calculators:
                    texts = stats["greedy_texts"]
                    values = calculator(dependencies=stats, texts=texts, model=None)
                    stats.update(values)

                man.stats = stats

                new_estimations = {}
                for estimator in estimators:
                    key = ('sequence', str(estimator))
                    old_values = old_estimations[key]
                    values = estimator(stats)
                    new_estimations[key] = values

                man.estimations = new_estimations

                metrics = deepcopy(man.metrics)

                man.ue_metrics = ue_metrics
                man.eval_ue()
                man.stats = {}

                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                man.save_path = os.path.join(out_dir, f"{model}_{dataset}.man")
                man.save()

if __name__ == '__main__':
    main()
