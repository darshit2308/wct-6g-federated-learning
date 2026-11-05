import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from experiment_runner import ExperimentConfig, run_experiment, MethodSpec


def run_threshold_sensitivity(base_config, thresholds=[0.45, 0.50, 0.55, 0.58, 0.62, 0.68, 0.72], seeds=[42, 52]):
    print("\n--- Running Selection Threshold Sensitivity Analysis (tau in [0.45, 0.72]) ---")
    results = []
    for tau in thresholds:
        print(f"Testing threshold tau = {tau:.2f}...")
        accuracies = []
        fairnesses = []
        macro_f1s = []
        for seed in seeds:
            config = ExperimentConfig(
                dataset_name=base_config.dataset_name,
                num_clients=base_config.num_clients,
                num_edges=base_config.num_edges,
                clients_per_round=base_config.clients_per_round,
                num_rounds=base_config.num_rounds,
                local_epochs=base_config.local_epochs,
                seed=seed,
                quiet=True,
            )
            # Create method with this threshold
            method = MethodSpec(
                name=f"proposed_tau_{tau:.2f}",
                topology="hierarchical",
                selection_strategy="fairness_aware",
                aggregation_strategy="quality_weighted",
                use_outlier_filter=True,
                fairness_temperature=1.0,
                description=f"Proposed method with selection threshold tau={tau:.2f}",
            )
            # Temporarily pass threshold to edge server via method if needed or monkey-patch
            from edge_server import EdgeServer
            original_init = EdgeServer.__init__
            def patched_init(self, *args, **kwargs):
                kwargs["selection_threshold"] = tau
                original_init(self, *args, **kwargs)
            EdgeServer.__init__ = patched_init

            try:
                res = run_experiment(config, method)
                s = res["summary"]
                accuracies.append(s["final_accuracy"])
                fairnesses.append(s["final_selection_fairness"])
                macro_f1s.append(s["final_macro_f1"])
            finally:
                EdgeServer.__init__ = original_init

        results.append({
            "threshold": tau,
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "fairness_mean": float(np.mean(fairnesses)),
            "fairness_std": float(np.std(fairnesses)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
        })

    df = pd.DataFrame(results)
    return df


def plot_sensitivity(df, output_path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel(r"Selection Desirability Threshold ($\tau$)", fontsize=12)
    ax1.set_ylabel("Global Accuracy", color=color, fontsize=12)
    ax1.errorbar(
        df["threshold"], df["accuracy_mean"], yerr=df["accuracy_std"],
        marker="o", color=color, capsize=4, label="Accuracy"
    )
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Jain's Fairness Index", color=color, fontsize=12)
    ax2.errorbar(
        df["threshold"], df["fairness_mean"], yerr=df["fairness_std"],
        marker="s", color=color, linestyle="--", capsize=4, label="Fairness"
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(r"Parameter Sensitivity Across Threshold $\tau \in [0.45, 0.72]$", fontsize=13)
    fig.tight_layout()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Sensitivity plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sensitivity Analysis for Hierarchical FL")
    parser.add_argument("--dataset", default="digits", choices=["digits", "fashion_mnist", "synthetic"])
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-edges", type=int, default=2)
    parser.add_argument("--clients-per-round", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=6)
    parser.add_argument("--output-dir", default="results_sensitivity")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = ExperimentConfig(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        num_edges=args.num_edges,
        clients_per_round=args.clients_per_round,
        num_rounds=args.num_rounds,
        local_epochs=2,
    )

    df = run_threshold_sensitivity(base_config)
    csv_path = out_dir / "threshold_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sensitivity table to {csv_path}:\n", df.to_string(index=False))

    plot_path = out_dir / "threshold_sensitivity.png"
    plot_sensitivity(df, plot_path)


if __name__ == "__main__":
    main()
