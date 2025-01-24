import argparse
import pathlib
import os
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.estimators import *
from lm_polygraph.ue_metrics import *
from lm_polygraph.stat_calculators import *
from tqdm import tqdm

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
    out_dir = 'sample_metric_mans/best_sample_with_greedy_enriched'

    estimators = [
        #SampledMaximumSequenceProbability(),
        SampledMaximumSequenceProbability(sample_strategy="best"),
        #SampledMaximumSequenceProbability(sample_strategy="best_normalized"),
        #SampledPerplexity(),
        SampledPerplexity(sample_strategy="best"),
        #SampledPerplexity(sample_strategy="best_normalized"),
        #SampledTokenSAR(),
        SampledTokenSAR(sample_strategy="best"),
        #SampledTokenSAR(sample_strategy="best_normalized"),
        #SampledMeanTokenEntropy(),
        SampledMeanTokenEntropy(sample_strategy="best"),
        #SampledMeanTokenEntropy(sample_strategy="best_normalized"),
#
        SAR(),
        TokenSAR(),
#
        #MaxprobGSU(exp=True),
        #PPLGSU(exp=True),
        #TokenSARGSU(exp=True),
        #MaxprobGSU(),
        #PPLGSU(),
        #TokenSARGSU(),
        #MTEGSU(),
#
        #AveMaxprob(),
        #AvePPL(),
        #AveTokenSAR(),
        #AveMTE(),
#
        #SemanticAveMaxprob(exp=True),
        #SemanticAvePPL(exp=True),
        #SemanticAveTokenSAR(exp=True),
        #SemanticAveMaxprob(),
        #SemanticAvePPL(),
        #SemanticAveTokenSAR(),
        #SemanticAveMTE(),
        #SemanticAveMaxprob(exp=True, sample_strategy="best"),
        #SemanticAvePPL(exp=True, sample_strategy="best"),
        #SemanticAveTokenSAR(exp=True, sample_strategy="best"),
        #SemanticAveMaxprob(sample_strategy="best"),
        #SemanticAvePPL(sample_strategy="best"),
        #SemanticAveTokenSAR(sample_strategy="best"),
        #SemanticAveMTE(sample_strategy="best"),
        #SemanticAveMaxprob(exp=True, sample_strategy="best_normalized"),
        #SemanticAvePPL(exp=True, sample_strategy="best_normalized"),
        #SemanticAveTokenSAR(exp=True, sample_strategy="best_normalized"),
        #SemanticAveMaxprob(sample_strategy="best_normalized"),
        #SemanticAvePPL(sample_strategy="best_normalized"),
        #SemanticAveTokenSAR(sample_strategy="best_normalized"),
        #SemanticAveMTE(sample_strategy="best_normalized"),
#
        #SemanticMedianMaxprob(exp=True),
        #SemanticMedianPPL(exp=True),
        #SemanticMedianTokenSAR(exp=True),
        #SemanticMedianMaxprob(),
        #SemanticMedianPPL(),
        #SemanticMedianTokenSAR(),
        #SemanticMedianMTE(),
        #SemanticMedianMaxprob(exp=True, sample_strategy="best"),
        #SemanticMedianPPL(exp=True, sample_strategy="best"),
        #SemanticMedianTokenSAR(exp=True, sample_strategy="best"),
        #SemanticMedianMaxprob(sample_strategy="best"),
        #SemanticMedianPPL(sample_strategy="best"),
        #SemanticMedianTokenSAR(sample_strategy="best"),
        #SemanticMedianMTE(sample_strategy="best"),
        #SemanticMedianMaxprob(exp=True, sample_strategy="best_normalized"),
        #SemanticMedianPPL(exp=True, sample_strategy="best_normalized"),
        #SemanticMedianTokenSAR(exp=True, sample_strategy="best_normalized"),
        #SemanticMedianMaxprob(sample_strategy="best_normalized"),
        #SemanticMedianPPL(sample_strategy="best_normalized"),
        #SemanticMedianTokenSAR(sample_strategy="best_normalized"),
        #SemanticMedianMTE(sample_strategy="best_normalized"),
#
        CEDegMat(),
#
        #GreedySemanticAveMaxprobAveSimilarity(exp=True),
        #GreedySemanticAvePPLAveSimilarity(exp=True),
        #GreedySemanticAveTokenSARAveSimilarity(exp=True),
        #GreedySemanticEnrichedPPLAveDissimilarity(exp=True),
        #GreedySemanticEnrichedTokenSARAveDissimilarity(exp=True),
        #GreedySemanticEnrichedMaxprobAveDissimilarity(exp=True),
        GreedySemanticAveMaxprobAveSimilarity(),
        GreedySemanticAvePPLAveSimilarity(),
        GreedySemanticAveTokenSARAveSimilarity(),
        GreedySemanticEnrichedPPLAveDissimilarity(),
        GreedySemanticEnrichedTokenSARAveDissimilarity(),
        GreedySemanticEnrichedMaxprobAveDissimilarity(),
        GreedySemanticAveMTEAveSimilarity(),
        GreedySemanticEnrichedMTEAveDissimilarity(),
#
        GreedySumSemanticMaxprob(),
        GreedySumSemanticPPL(),
        GreedySumSemanticMTE(),
        GreedySupSumSemanticMaxprob(),
        GreedySupSumSemanticPPL(),
        GreedySupSumSemanticMTE(),
        GreedySupSumSemanticMaxprob(alpha=.1),
        GreedySupSumSemanticPPL(alpha=.1),
        GreedySupSumSemanticMTE(alpha=.1),
        GreedySupSumSemanticMaxprob(alpha=.3),
        GreedySupSumSemanticPPL(alpha=.3),
        GreedySupSumSemanticMTE(alpha=.3),
        GreedySupSumSemanticMaxprob(alpha=.5),
        GreedySupSumSemanticPPL(alpha=.5),
        GreedySupSumSemanticMTE(alpha=.5),
        GreedySupSumSemanticMaxprob(alpha=.7),
        GreedySupSumSemanticPPL(alpha=.7),
        GreedySupSumSemanticMTE(alpha=.7),
        GreedySupSumSemanticMaxprob(alpha=1.2),
        GreedySupSumSemanticPPL(alpha=1.2),
        GreedySupSumSemanticMTE(alpha=1.2),
        GreedySupSumSemanticMaxprob(alpha=1.5),
        GreedySupSumSemanticPPL(alpha=1.5),
        GreedySupSumSemanticMTE(alpha=1.5),
#
        #SemanticAveMaxprobAveSimilarity(exp=True),
        #SemanticAvePPLAveSimilarity(exp=True),
        #SemanticAveTokenSARAveSimilarity(exp=True),
        #SemanticEnrichedPPLAveDissimilarity(exp=True),
        #SemanticEnrichedTokenSARAveDissimilarity(exp=True),
        #SemanticEnrichedMaxprobAveDissimilarity(exp=True),
        #SemanticAveMaxprobAveSimilarity(),
        #SemanticAvePPLAveSimilarity(),
        #SemanticAveTokenSARAveSimilarity(),
        #SemanticEnrichedPPLAveDissimilarity(),
        #SemanticEnrichedTokenSARAveDissimilarity(),
        #SemanticEnrichedMaxprobAveDissimilarity(),
        #SemanticAveMTEAveSimilarity(),
        #SemanticEnrichedMTEAveDissimilarity(),
#
        #SumSemanticMaxprob(),
        #SumSemanticPPL(),
        #SumSemanticMTE(),
#
        #SemanticAveMaxprobAveSimilarity(exp=True, sample_strategy="best"),
        #SemanticAvePPLAveSimilarity(exp=True, sample_strategy="best"),
        #SemanticAveTokenSARAveSimilarity(exp=True, sample_strategy="best"),
        #SemanticEnrichedPPLAveDissimilarity(exp=True, sample_strategy="best"),
        #SemanticEnrichedTokenSARAveDissimilarity(exp=True, sample_strategy="best"),
        #SemanticEnrichedMaxprobAveDissimilarity(exp=True, sample_strategy="best"),
        SemanticAveMaxprobAveSimilarity(sample_strategy="best"),
        SemanticAvePPLAveSimilarity(sample_strategy="best"),
        SemanticAveTokenSARAveSimilarity(sample_strategy="best"),
        SemanticEnrichedPPLAveDissimilarity(sample_strategy="best"),
        SemanticEnrichedTokenSARAveDissimilarity(sample_strategy="best"),
        SemanticEnrichedMaxprobAveDissimilarity(sample_strategy="best"),
        SemanticAveMTEAveSimilarity(sample_strategy="best"),
        SemanticEnrichedMTEAveDissimilarity(sample_strategy="best"),
#
        SumSemanticMaxprob(sample_strategy="best"),
        SumSemanticPPL(sample_strategy="best"),
        SumSemanticMTE(sample_strategy="best"),
        SupSumSemanticMaxprob(sample_strategy="best"),
        SupSumSemanticPPL(sample_strategy="best"),
        SupSumSemanticMTE(sample_strategy="best"),
        SupSumSemanticMaxprob(sample_strategy="best", alpha=.1),
        SupSumSemanticPPL(sample_strategy="best", alpha=.1),
        SupSumSemanticMTE(sample_strategy="best", alpha=.1),
        SupSumSemanticMaxprob(sample_strategy="best", alpha=.3),
        SupSumSemanticPPL(sample_strategy="best", alpha=.3),
        SupSumSemanticMTE(sample_strategy="best", alpha=.3),
        SupSumSemanticMaxprob(sample_strategy="best", alpha=.5),
        SupSumSemanticPPL(sample_strategy="best", alpha=.5),
        SupSumSemanticMTE(sample_strategy="best", alpha=.5),
        SupSumSemanticMaxprob(sample_strategy="best", alpha=.7),
        SupSumSemanticPPL(sample_strategy="best", alpha=.7),
        SupSumSemanticMTE(sample_strategy="best", alpha=.7),
        SupSumSemanticMaxprob(sample_strategy="best", alpha=1.2),
        SupSumSemanticPPL(sample_strategy="best", alpha=1.2),
        SupSumSemanticMTE(sample_strategy="best", alpha=1.2),
        SupSumSemanticMaxprob(sample_strategy="best", alpha=1.5),
        SupSumSemanticPPL(sample_strategy="best", alpha=1.5),
        SupSumSemanticMTE(sample_strategy="best", alpha=1.5),
#
        #SemanticAveMaxprobAveSimilarity(exp=True, sample_strategy="best_normalized"),
        #SemanticAvePPLAveSimilarity(exp=True, sample_strategy="best_normalized"),
        #SemanticAveTokenSARAveSimilarity(exp=True, sample_strategy="best_normalized"),
        #SemanticEnrichedPPLAveDissimilarity(exp=True, sample_strategy="best_normalized"),
        #SemanticEnrichedTokenSARAveDissimilarity(exp=True, sample_strategy="best_normalized"),
        #SemanticEnrichedMaxprobAveDissimilarity(exp=True, sample_strategy="best_normalized"),
        #SemanticAveMaxprobAveSimilarity(sample_strategy="best_normalized"),
        #SemanticAvePPLAveSimilarity(sample_strategy="best_normalized"),
        #SemanticAveTokenSARAveSimilarity(sample_strategy="best_normalized"),
        #SemanticEnrichedPPLAveDissimilarity(sample_strategy="best_normalized"),
        #SemanticEnrichedTokenSARAveDissimilarity(sample_strategy="best_normalized"),
        #SemanticEnrichedMaxprobAveDissimilarity(sample_strategy="best_normalized"),
        #SemanticAveMTEAveSimilarity(sample_strategy="best_normalized"),
        #SemanticEnrichedMTEAveDissimilarity(sample_strategy="best_normalized"),
#
        #SumSemanticMaxprob(sample_strategy="best_normalized"),
        #SumSemanticPPL(sample_strategy="best_normalized"),
        #SumSemanticMTE(sample_strategy="best_normalized"),
#
        GreedyAveDissimilarity(),
        #AveDissimilarity(),
        AveDissimilarity(sample_strategy="best"),
        #AveDissimilarity(sample_strategy="best_normalized"),
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

            for calculator in stat_calculators:
                texts = stats["greedy_texts"]
                values = calculator(dependencies=stats, texts=texts, model=None)
                stats.update(values)

            for estimator in estimators:
                values = estimator(stats)
                man.estimations[('sequence', str(estimator))] = values

            man.stats = stats

            man.ue_metrics = ue_metrics

            man.eval_ue()
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            man.save_path = os.path.join(out_dir, f"{model}_{dataset}.man")
            man.save()

if __name__ == '__main__':
    main()
