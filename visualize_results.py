import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_graphs(results_dir):
    results_dir = Path(results_dir)
    round_metrics_path = results_dir / "round_metrics.csv"
    summary_path = results_dir / "summary.csv"

    if not round_metrics_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Expected real experiment outputs in the results directory. "
            "Run `python main.py --mode suite` first."
        )

    round_metrics = pd.read_csv(round_metrics_path)
    summary = pd.read_csv(summary_path)
    client_path = results_dir / "client_participation.csv"
    client_participation = pd.read_csv(client_path) if client_path.exists() else None

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    for method_name, method_frame in round_metrics.groupby("method"):
        ax.plot(
            method_frame["round"],
            method_frame["accuracy_mean"],
            marker="o",
            label=method_name,
        )
    ax.set_title("Global Accuracy by Round")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_by_round.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method_name, method_frame in round_metrics.groupby("method"):
        ax.plot(
            method_frame["round"],
            method_frame["loss_mean"],
            marker="s",
            label=method_name,
        )
    ax.set_title("Global Loss by Round")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_by_round.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["method"], summary["total_cloud_uploads_mean"], color="#1f77b4")
    ax.set_title("Total Cloud Uploads by Method")
    ax.set_ylabel("Uploads")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots_dir / "cloud_uploads_by_method.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["method"], summary["total_payload_bytes_mean"], color="#2ca02c")
    ax.set_title("Total Communication Bytes by Method")
    ax.set_ylabel("Bytes")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots_dir / "payload_bytes_by_method.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["method"], summary["mean_round_latency_proxy_ms_mean"], color="#d62728")
    ax.set_title("Mean Latency Proxy by Method")
    ax.set_ylabel("Latency Proxy (ms)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots_dir / "latency_by_method.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["method"], summary["final_selection_fairness_mean"], color="#9467bd")
    ax.set_title("Selection Fairness by Method")
    ax.set_ylabel("Jain Fairness")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots_dir / "fairness_by_method.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["method"], summary["attack_detection_rate_mean"], color="#8c564b")
    ax.set_title("Attack Detection Rate by Method")
    ax.set_ylabel("Detection Rate")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots_dir / "attack_detection_by_method.png", dpi=200)
    plt.close(fig)

    if client_participation is not None:
        fig, ax = plt.subplots(figsize=(11, 6))
        for method_name, method_frame in client_participation.groupby("method"):
            method_frame = method_frame.sort_values("selection_count_mean", ascending=False).head(10)
            ax.plot(
                method_frame["client_id"],
                method_frame["selection_count_mean"],
                marker="o",
                label=method_name,
            )
        ax.set_title("Top Client Participation Counts")
        ax.set_xlabel("Client")
        ax.set_ylabel("Mean Selections")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "client_participation_profile.png", dpi=200)
        plt.close(fig)

    print(f"Saved real benchmark plots to {plots_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot actual benchmark results from the exported CSV files."
    )
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_graphs(args.results_dir)
