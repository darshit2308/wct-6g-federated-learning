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

from experiment_runner import ExperimentConfig, run_experiment, MethodSpec, DEFAULT_METHODS


def run_pareto_sweep(base_config, temperatures=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], seed=42):
    print("\n--- Generating Pareto Frontier Sweep (Accuracy vs. Jain Fairness) ---")
    points = []

    # 1. Sweep proposed method across fairness temperatures
    for T in temperatures:
        print(f"Running Proposed with Fairness Temperature T = {T:.2f}...")
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
        method = MethodSpec(
            name=f"proposed_T_{T:.2f}",
            topology="hierarchical",
            selection_strategy="fairness_aware",
            aggregation_strategy="quality_weighted",
            use_outlier_filter=True,
            fairness_temperature=T,
            description=f"Proposed with T_fair={T:.2f}",
        )
        res = run_experiment(config, method)
        s = res["summary"]
        points.append({
            "name": f"Proposed (T={T:.2f})",
            "category": "Proposed (Pareto sweep)",
            "accuracy": s["final_accuracy"],
            "fairness": s["final_selection_fairness"],
            "cloud_uploads": s["total_cloud_uploads"],
            "macro_f1": s["final_macro_f1"],
            "gini": s["final_participation_gini"],
        })

    # 2. Add key baselines for visual Pareto comparison
    baselines_to_test = [
        "flat_fedavg_random",
        "hierarchical_random_fedavg",
        "hierarchical_intelligent_no_fairness",
        "hierarchical_direct_score",
    ]
    for b_name in baselines_to_test:
        print(f"Running Baseline: {b_name}...")
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
        res = run_experiment(config, DEFAULT_METHODS[b_name])
        s = res["summary"]
        points.append({
            "name": b_name,
            "category": "Baseline",
            "accuracy": s["final_accuracy"],
            "fairness": s["final_selection_fairness"],
            "cloud_uploads": s["total_cloud_uploads"],
            "macro_f1": s["final_macro_f1"],
            "gini": s["final_participation_gini"],
        })

    return pd.DataFrame(points)


def plot_pareto(df, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    proposed_pts = df[df["category"] == "Proposed (Pareto sweep)"].sort_values("fairness")
    baseline_pts = df[df["category"] == "Baseline"]

    # Plot proposed Pareto curve
    ax.plot(
        proposed_pts["fairness"], proposed_pts["accuracy"],
        marker="o", color="#1f77b4", linewidth=2.5, markersize=8,
        label="Proposed Adaptive Pareto Frontier", zorder=3
    )

    # Annotate temperatures
    for _, row in proposed_pts.iterrows():
        t_label = row["name"].replace("Proposed ", "")
        ax.annotate(
            t_label,
            (row["fairness"], row["accuracy"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#0d47a1",
        )

    # Plot baselines
    colors = {"flat_fedavg_random": "#e377c2", "hierarchical_random_fedavg": "#ff7f0e",
              "hierarchical_intelligent_no_fairness": "#d62728", "hierarchical_direct_score": "#2ca02c"}
    markers = {"flat_fedavg_random": "s", "hierarchical_random_fedavg": "^",
               "hierarchical_intelligent_no_fairness": "D", "hierarchical_direct_score": "X"}

    for _, row in baseline_pts.iterrows():
        b_name = row["name"]
        ax.scatter(
            row["fairness"], row["accuracy"],
            color=colors.get(b_name, "#7f7f7f"),
            marker=markers.get(b_name, "o"),
            s=120,
            label=f"Baseline: {b_name}",
            zorder=4,
            edgecolors="black",
        )

    ax.set_xlabel("Participation Fairness (Jain's Index)", fontsize=12)
    ax.set_ylabel("Global Model Accuracy", fontsize=12)
    ax.set_title("Multi-Objective Pareto Frontier: Accuracy vs. Participation Fairness", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    print(f"Pareto frontier plot successfully saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-objective Pareto Frontier Generator")
    parser.add_argument("--dataset", default="digits", choices=["digits", "fashion_mnist", "synthetic"])
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-edges", type=int, default=2)
    parser.add_argument("--clients-per-round", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=6)
    parser.add_argument("--output-dir", default="results_pareto")
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

    df = run_pareto_sweep(base_config)
    csv_path = out_dir / "pareto_points.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPareto points table exported to {csv_path}:\n", df.to_string(index=False))

    plot_path = out_dir / "pareto_frontier.png"
    plot_pareto(df, plot_path)


if __name__ == "__main__":
    main()
