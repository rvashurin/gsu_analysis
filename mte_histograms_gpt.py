import os
import matplotlib.pyplot as plt
import argparse
from lm_polygraph.utils.manager import UEManager


def main(do_sample: bool):
    # Define models and datasets
    models = ["mistral7b", "llama8b"]
    datasets = [
        "trivia", "mmlu", "coqa", "gsm8k_cot",
        "xsum", "wmt14_fren", "wmt19_deen",
        "wmt14_enfr", "wmt19_ende"
    ]

    # Define uncertainty methods to extract
#    if do_sample:
#        methods = ["SampledMeanTokenEntropy", "MTESAR", "MTEGSU"]
#    else:
#        methods = ["MeanTokenEntropy", "MTESAR", "MTEGSU"]
    methods = ["SampledPPL", "PPLGSU", "PPLGSUexp"]

    # Determine the base directory for managers and output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_input_dir = "sample_metric_mans/log_exp" if do_sample else "greedy_metric_mans/log_exp"
    base_output_dir = "gsu_exp_histograms/sample" if do_sample else "gsu_exp_histograms/greedy"

    # Ensure output directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    # Loop through each model and dataset combination
    for model in models:
        for dataset in datasets:
            # Construct manager file path
            manager_filename = f"{model}_{dataset}.man"
            manager_path = os.path.join(script_dir, base_input_dir, manager_filename)

            # Check if the manager file exists
            if not os.path.exists(manager_path):
                print(f"Manager file not found: {manager_path}")
                continue

            # Load the manager
            manager = UEManager.load(manager_path)

            # Extract uncertainty values for specified methods
            estimations = manager.estimations
            data = {method: estimations.get(('sequence', method), []) for method in methods}

            # Plot histograms for the three methods
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"{model} - {dataset}", fontsize=16)

            for ax, method in zip(axes, methods):
                ax.hist(data[method], bins=30, edgecolor='black', alpha=0.7)
                ax.set_title(method)
                ax.set_xlabel("Uncertainty Value")
                ax.set_ylabel("Frequency")

            # Save the plot as a PNG file
            output_filename = f"{model}_{dataset}_histograms.png"
            output_path = os.path.join(script_dir, base_output_dir, output_filename)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to include title
            plt.savefig(output_path)
            plt.close()
            print(f"Histograms saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate uncertainty histograms.")
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="If set, use sample_metric_mans directory for managers; otherwise, use greedy_metric_mans/mte_fixed_ave_appended."
    )
    args = parser.parse_args()
    main(do_sample=args.do_sample)
