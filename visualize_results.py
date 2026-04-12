import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError(
        "matplotlib is required to generate plots. Install the project requirements first."
    ) from exc


def _line_plot_with_band(frame, metric, metric_label, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for method_name, method_frame in frame.groupby("method"):
        method_frame = method_frame.sort_values("round")
        x_values = method_frame["round"]
        y_values = method_frame[f"{metric}_mean"]
        y_std = method_frame.get(f"{metric}_std")
        ax.plot(x_values, y_values, marker="o", label=method_name)
        if y_std is not None:
            ax.fill_between(
                x_values,
                y_values - y_std,
                y_values + y_std,
                alpha=0.15,
            )
    ax.set_title(title)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(metric_label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _bar_plot(summary, metric, metric_label, title, output_path, color):
    fig, ax = plt.subplots(figsize=(10, 6))
    errors = summary.get(f"{metric}_std")
    ax.bar(
        summary["method"],
        summary[f"{metric}_mean"],
        yerr=errors,
        capsize=4 if errors is not None else 0,
        color=color,
    )
    ax.set_title(title)
    ax.set_ylabel(metric_label)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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

    _line_plot_with_band(
        round_metrics,
        metric="accuracy",
        metric_label="Accuracy",
        title="Global Accuracy by Round",
        output_path=plots_dir / "accuracy_by_round.png",
    )
    _line_plot_with_band(
        round_metrics,
        metric="loss",
        metric_label="Loss",
        title="Global Loss by Round",
        output_path=plots_dir / "loss_by_round.png",
    )
    _line_plot_with_band(
        round_metrics,
        metric="macro_f1",
        metric_label="Macro-F1",
        title="Global Macro-F1 by Round",
        output_path=plots_dir / "macro_f1_by_round.png",
    )
    _line_plot_with_band(
        round_metrics,
        metric="balanced_accuracy",
        metric_label="Balanced Accuracy",
        title="Balanced Accuracy by Round",
        output_path=plots_dir / "balanced_accuracy_by_round.png",
    )

    _bar_plot(
        summary,
        metric="total_cloud_uploads",
        metric_label="Uploads",
        title="Total Cloud Uploads by Method",
        output_path=plots_dir / "cloud_uploads_by_method.png",
        color="#1f77b4",
    )
    _bar_plot(
        summary,
        metric="total_payload_bytes",
        metric_label="Bytes",
        title="Total Communication Bytes by Method",
        output_path=plots_dir / "payload_bytes_by_method.png",
        color="#2ca02c",
    )
    _bar_plot(
        summary,
        metric="mean_round_latency_proxy_ms",
        metric_label="Latency Proxy (ms)",
        title="Mean Latency Proxy by Method",
        output_path=plots_dir / "latency_by_method.png",
        color="#d62728",
    )
    _bar_plot(
        summary,
        metric="final_selection_fairness",
        metric_label="Jain Fairness",
        title="Selection Fairness by Method",
        output_path=plots_dir / "fairness_by_method.png",
        color="#9467bd",
    )
    _bar_plot(
        summary,
        metric="final_participation_gini",
        metric_label="Gini Coefficient",
        title="Participation Inequality by Method",
        output_path=plots_dir / "participation_gini_by_method.png",
        color="#bcbd22",
    )
    _bar_plot(
        summary,
        metric="attack_detection_rate",
        metric_label="Detection Rate",
        title="Attack Detection Rate by Method",
        output_path=plots_dir / "attack_detection_by_method.png",
        color="#8c564b",
    )
    _bar_plot(
        summary,
        metric="benign_retention_rate",
        metric_label="Retention Rate",
        title="Benign Retention by Method",
        output_path=plots_dir / "benign_retention_by_method.png",
        color="#17becf",
    )

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
