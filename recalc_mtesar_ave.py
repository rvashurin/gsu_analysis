import argparse
import pathlib
import os
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.estimators import *
from lm_polygraph.ue_metrics import *
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    # boolean argument do_sample with default value of False
    parser.add_argument('--do_sample', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()

    # Define models and datasets
    models = ["mistral7b", "llama8b"]
    datasets = [
        "trivia", "mmlu", "coqa", "gsm8k_cot", 
        "xsum", "wmt14_fren", "wmt19_deen", 
        "wmt14_enfr", "wmt19_ende"
    ]

    # Get the directory containing this script
    if args.do_sample:
        script_dir = 'sample_metric_mans/no_log'
        out_dir = 'sample_metric_mans/log_exp' 
    else:
        script_dir = 'greedy_metric_mans/no_log'
        out_dir = 'greedy_metric_mans/log_exp' 

    estimators = [
        MaxprobGSU(exp=True),
        PPLGSU(exp=True),
        TokenSARGSU(exp=True),
        MaxprobGSU(),
        PPLGSU(),
        TokenSARGSU(),
        MTEGSU(),
        #AveMaxprob(),
        #AvePPL(),
        #AveTokenSAR(),
        #AveMTE(),
        SemanticAveMaxprob(exp=True),
        SemanticAvePPL(exp=True),
        SemanticAveTokenSAR(exp=True),
        SemanticAveMTE(),
        CEDegMat(),
    ]

    ue_metrics = [
        PredictionRejectionArea(),
        PredictionRejectionArea(max_rejection=0.5),
    ]

    # Loop through each model and dataset combination
    for model in tqdm(models):
        for dataset in tqdm(datasets):
            # Construct manager file path
            manager_filename = f"{model}_{dataset}.man"
            manager_path = os.path.join(script_dir, manager_filename)

            man = UEManager.load(manager_path)

            stats = man.stats

            for estimator in estimators:
                values = estimator(stats)

                man.estimations[('sequence', str(estimator))] = values

            man.ue_metrics = ue_metrics

            man.eval_ue()
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            man.save(os.path.join(out_dir, f"{model}_{dataset}.man"))

if __name__ == '__main__':
    main()
