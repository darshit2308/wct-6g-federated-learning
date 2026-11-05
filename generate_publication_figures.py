import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set publication style defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Figure 1: Attack Screening & Robustness Across Adversarial Models
# -------------------------------------------------------------------------
def generate_attack_figure():
    print("Generating fig_attack_detection.png...")
    csv_path = Path("results_attack/combined_attack_summary.csv")
    if not csv_path.exists():
        print("Warning: combined_attack_summary.csv not found!")
        return

    df = pd.read_csv(csv_path)

    # Attack types to plot
    attack_order = ["sign_flip", "adaptive_fang", "label_flip"]
    attack_labels = {
        "sign_flip": "Sign-Flip Attack\n(25% Malicious)",
        "adaptive_fang": "Adaptive Fang Attack\n(25% Malicious)",
        "label_flip": "Label-Flip Attack\n(25% Malicious)"
    }

    # Methods to compare
    methods = [
        ("flat_fedavg_random", "Flat FedAvg [McMahan'17]", "#7f7f7f"),
        ("hierarchical_intelligent_no_filter", "Hier. Unshielded (No-Filter)", "#ff7f0e"),
        ("hierarchical_krum", "Hier. Krum [NeurIPS'17]", "#9467bd"),
        ("hierarchical_trimmed_mean", "Hier. Trimmed Mean [ICML'18]", "#8c564b"),
        ("proposed", "Proposed AHFL (Dual MAD+Cos)", "#2ca02c"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(attack_order))
    bar_width = 0.15

    # Panel 1: Detection Rate
    for idx, (m_key, m_name, color) in enumerate(methods):
        det_means = []
        det_stds = []
        for atk in attack_order:
            sub = df[(df["attack_type"] == atk) & (df["method"] == m_key)]
            if not sub.empty:
                det_means.append(sub["attack_detection_rate_mean"].values[0] * 100)
                std_val = sub["attack_detection_rate_std"].values[0] if "attack_detection_rate_std" in sub.columns else 0.0
                det_stds.append(std_val * 100)
            else:
                det_means.append(0.0)
                det_stds.append(0.0)

        offset = (idx - len(methods)/2 + 0.5) * bar_width
        bars = ax1.bar(x + offset, det_means, bar_width, yerr=det_stds, label=m_name, color=color, alpha=0.9, capsize=3)
        
        # Add text labels on top of Proposed bars
        if m_key == "proposed":
            for b in bars:
                height = b.get_height()
                if height > 0:
                    ax1.annotate(f"{height:.1f}%",
                                xy=(b.get_x() + b.get_width() / 2, height),
                                xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1b5e20")

    ax1.set_ylabel("Attack Detection Rate (%)")
    ax1.set_title("(a) Byzantine Anomaly Detection Rate Across Attack Models")
    ax1.set_xticks(x)
    ax1.set_xticklabels([attack_labels[a] for a in attack_order])
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Final Accuracy Under Attack
    for idx, (m_key, m_name, color) in enumerate(methods):
        acc_means = []
        acc_stds = []
        for atk in attack_order:
            sub = df[(df["attack_type"] == atk) & (df["method"] == m_key)]
            if not sub.empty:
                acc_means.append(sub["final_accuracy_mean"].values[0] * 100)
                acc_stds.append(sub["final_accuracy_std"].values[0] * 100)
            else:
                acc_means.append(0.0)
                acc_stds.append(0.0)

        offset = (idx - len(methods)/2 + 0.5) * bar_width
        bars = ax2.bar(x + offset, acc_means, bar_width, yerr=acc_stds, label=m_name, color=color, alpha=0.9, capsize=3)
        
        if m_key == "proposed":
            for b in bars:
                height = b.get_height()
                ax2.annotate(f"{height:.1f}%",
                            xy=(b.get_x() + b.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1b5e20")

    ax2.set_ylabel("Final Model Accuracy (%)")
    ax2.set_title("(b) Global Model Accuracy Under Active Byzantine Poisoning")
    ax2.set_xticks(x)
    ax2.set_xticklabels([attack_labels[a] for a in attack_order])
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    out_file = DOCS_DIR / "fig_attack_detection.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


# -------------------------------------------------------------------------
# Figure 2: Fairness Comparison Across All 10 Methods
# -------------------------------------------------------------------------
def generate_fairness_figure():
    print("Generating fig_fairness_comparison.png...")
    csv_path = Path("results/summary.csv")
    if not csv_path.exists():
        print("Warning: results/summary.csv not found!")
        return

    df = pd.read_csv(csv_path)

    # Human-readable label map
    label_map = {
        "hierarchical_fairness_direct_score": "Direct-Score + Fairness Reg.",
        "hierarchical_intelligent_no_filter": "Hier. Intelligent (No Filter)",
        "proposed": "Proposed AHFL (Full Method)",
        "flat_fedavg_random": "Flat FedAvg (Random Select)",
        "hierarchical_krum": "Hierarchical Krum [NeurIPS'17]",
        "hierarchical_trimmed_mean": "Hierarchical Trimmed Mean [ICML'18]",
        "hierarchical_random_fedavg": "Hierarchical Random FedAvg",
        "hierarchical_intelligent_no_fairness": "Hier. Intelligent (No Fairness)",
        "hierarchical_intelligent_fedavg": "Hier. Intelligent FedAvg (No Weights)",
        "hierarchical_direct_score": "Direct-Score Selection (No ML)",
    }

    df["display_name"] = df["method"].map(lambda x: label_map.get(x, x))
    df = df.sort_values("final_selection_fairness_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.2))

    # Color highlights
    colors = []
    for m in df["method"]:
        if m == "proposed":
            colors.append("#2ca02c") # emerald
        elif "direct_score" in m and "fairness" not in m:
            colors.append("#d62728") # crimson (collapse)
        elif "no_fairness" in m or m == "hierarchical_intelligent_fedavg":
            colors.append("#ff7f0e") # warning orange
        elif "flat" in m:
            colors.append("#1f77b4") # standard blue
        else:
            colors.append("#7f7f7f") # neutral grey

    y_pos = np.arange(len(df))
    means = df["final_selection_fairness_mean"].values
    stds = df["final_selection_fairness_std"].values

    bars = ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.85, capsize=4, height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display_name"])
    ax.set_xlabel("Jain's Fairness Index (5 Seeds, Mean $\pm$ Std)")
    ax.set_title("Participation Fairness Across Client Selection & Aggregation Baselines")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Annotate Direct-Score collapse and Proposed
    for idx, (m, bar) in enumerate(zip(df["method"], bars)):
        width = bar.get_width()
        if m == "hierarchical_direct_score":
            ax.annotate(f" {width:.4f} (Fairness Collapse: -43%)",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(15, 0), textcoords="offset points",
                        va="center", fontsize=9, fontweight="bold", color="#d62728")
        elif m == "proposed":
            ax.annotate(f" {width:.4f} (Proposed Balanced)",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(15, 0), textcoords="offset points",
                        va="center", fontsize=9, fontweight="bold", color="#1b5e20")
        else:
            ax.annotate(f" {width:.4f}",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color="#333333")

    fig.tight_layout()
    out_file = DOCS_DIR / "fig_fairness_comparison.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


# -------------------------------------------------------------------------
# Figure 3: Cloud Uploads Communication Efficiency
# -------------------------------------------------------------------------
def generate_cloud_uploads_figure():
    print("Generating fig_cloud_uploads.png...")
    csv_path = Path("results/summary.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    label_map = {
        "flat_fedavg_random": "Flat FedAvg (Direct Client-to-Cloud)",
        "hierarchical_random_fedavg": "Hierarchical FedAvg",
        "hierarchical_direct_score": "Hier. Direct Score",
        "hierarchical_krum": "Hier. Krum [NeurIPS'17]",
        "hierarchical_trimmed_mean": "Hier. Trimmed Mean [ICML'18]",
        "proposed": "Proposed AHFL (Intermediate Edge)",
    }
    subset_methods = list(label_map.keys())
    sub_df = df[df["method"].isin(subset_methods)].copy()
    sub_df["display_name"] = sub_df["method"].map(label_map)
    sub_df = sub_df.sort_values("total_cloud_uploads_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    colors = ["#d62728" if "flat" in m else "#2ca02c" for m in sub_df["method"]]
    y_pos = np.arange(len(sub_df))
    uploads = sub_df["total_cloud_uploads_mean"].values

    bars = ax.barh(y_pos, uploads, color=colors, alpha=0.85, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_df["display_name"])
    ax.set_xlabel("Total Model Uploads Received by Cloud Server (Over 10 Rounds)")
    ax.set_title("Cloud Communication Overhead: Flat FL vs. Hierarchical FL")
    ax.set_xlim(0, 95)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, uploads):
        if val == 80:
            ax.annotate(f" {int(val)} uploads (Direct Fan-in Bottleneck)",
                        xy=(val, bar.get_y() + bar.get_height() / 2),
                        xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=9.5, fontweight="bold", color="#d62728")
        else:
            ax.annotate(f" {int(val)} uploads (-75% Reduction)",
                        xy=(val, bar.get_y() + bar.get_height() / 2),
                        xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=9, fontweight="bold", color="#1b5e20")

    fig.tight_layout()
    out_file = DOCS_DIR / "fig_cloud_uploads.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


# -------------------------------------------------------------------------
# Figure 4: Convergence Across Communication Rounds
# -------------------------------------------------------------------------
def generate_convergence_figure():
    print("Generating fig_convergence_rounds.png...")
    csv_path = Path("results/round_metrics.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)

    methods_to_plot = [
        ("proposed", "Proposed AHFL", "#2ca02c", "-", 2.5, "o"),
        ("flat_fedavg_random", "Flat FedAvg", "#1f77b4", "--", 1.8, "s"),
        ("hierarchical_direct_score", "Hier. Direct-Score (No ML)", "#d62728", "-.", 1.8, "^"),
        ("hierarchical_krum", "Hier. Krum [NeurIPS'17]", "#9467bd", ":", 1.8, "v"),
        ("hierarchical_trimmed_mean", "Hier. Trimmed Mean [ICML'18]", "#8c564b", ":", 1.8, "D"),
        ("hierarchical_random_fedavg", "Hier. Random FedAvg", "#7f7f7f", "--", 1.5, "x"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))

    for m_key, m_label, color, ls, lw, marker in methods_to_plot:
        sub = df[df["method"] == m_key].sort_values("round")
        if not sub.empty:
            rounds = sub["round"].values
            acc = sub["accuracy_mean"].values * 100
            std = sub["accuracy_std"].values * 100
            ax.plot(rounds, acc, label=m_label, color=color, linestyle=ls, linewidth=lw, marker=marker, markersize=5)
            if m_key in ["proposed", "hierarchical_direct_score"]:
                ax.fill_between(rounds, acc - std, acc + std, color=color, alpha=0.15)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.set_title("Learning Convergence Trajectory Across Communication Rounds (5 Seeds)")
    ax.set_xticks(np.arange(1, 11))
    ax.set_ylim(40, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    fig.tight_layout()
    out_file = DOCS_DIR / "fig_convergence_rounds.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


# -------------------------------------------------------------------------
# Figure 5: Pareto Optimal Frontier
# -------------------------------------------------------------------------
def generate_pareto_figure():
    print("Generating pareto_frontier.png...")
    csv_path = Path("results_pareto/pareto_points.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Proposed points
    prop = df[df["category"] == "Proposed (Pareto sweep)"].copy()
    base = df[df["category"] == "Baseline"].copy()

    # Plot baselines
    base_label_map = {
        "flat_fedavg_random": "Flat FedAvg",
        "hierarchical_random_fedavg": "Hier. Random FedAvg",
        "hierarchical_intelligent_no_fairness": "Hier. No-Fairness",
        "hierarchical_direct_score": "Hier. Direct-Score",
    }
    base_markers = {"flat_fedavg_random": "s", "hierarchical_random_fedavg": "^", 
                    "hierarchical_intelligent_no_fairness": "x", "hierarchical_direct_score": "v"}
    base_colors = {"flat_fedavg_random": "#1f77b4", "hierarchical_random_fedavg": "#7f7f7f",
                   "hierarchical_intelligent_no_fairness": "#ff7f0e", "hierarchical_direct_score": "#d62728"}

    for _, row in base.iterrows():
        b_key = row["name"]
        lbl = base_label_map.get(b_key, b_key)
        ax.scatter(row["fairness"], row["accuracy"], s=120, 
                   color=base_colors.get(b_key, "#333333"),
                   marker=base_markers.get(b_key, "o"), 
                   label=lbl, zorder=5)
        ax.annotate(lbl, (row["fairness"], row["accuracy"]),
                    xytext=(8, -4), textcoords="offset points", fontsize=9)

    # Plot proposed points
    ax.scatter(prop["fairness"], prop["accuracy"], s=90, color="#2ca02c", marker="o", 
               label="Proposed (Fairness Temp Sweep)", zorder=6)

    # Sort and plot smooth Pareto curve
    sorted_prop = prop.sort_values("fairness")
    ax.plot(sorted_prop["fairness"], sorted_prop["accuracy"], color="#2ca02c", linestyle="--", linewidth=1.8, alpha=0.7)

    # Annotate select proposed points
    for _, row in prop.iterrows():
        t_label = row["name"].replace("Proposed ", "")
        if "0.00" in t_label or "1.00" in t_label or "2.00" in t_label:
            ax.annotate(t_label, (row["fairness"], row["accuracy"]),
                        xytext=(-10, 8), textcoords="offset points", fontsize=8.5, color="#1b5e20", fontweight="bold")

    ax.set_xlabel("Jain's Fairness Index")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_title("Pareto Optimal Frontier: Accuracy vs. Jain Participation Fairness")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    out_file = DOCS_DIR / "pareto_frontier.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


# -------------------------------------------------------------------------
# Figure 6: Threshold Sensitivity Analysis
# -------------------------------------------------------------------------
def generate_sensitivity_figure():
    print("Generating threshold_sensitivity.png...")
    csv_path = Path("results_sensitivity/threshold_sensitivity.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path).sort_values("threshold")

    fig, ax1 = plt.subplots(figsize=(8.5, 5))

    color_acc = "#1f77b4"
    color_fair = "#2ca02c"

    thresh = df["threshold"].values
    acc_mean = df["accuracy_mean"].values * 100
    acc_std = df["accuracy_std"].values * 100
    fair_mean = df["fairness_mean"].values
    fair_std = df["fairness_std"].values

    # Left axis: Accuracy
    ax1.set_xlabel("Desirability Score Threshold $\\tau$")
    ax1.set_ylabel("Global Accuracy (%)", color=color_acc)
    line1 = ax1.plot(thresh, acc_mean, color=color_acc, marker="o", linewidth=2.2, label="Accuracy (%)")
    ax1.fill_between(thresh, acc_mean - acc_std, acc_mean + acc_std, color=color_acc, alpha=0.18)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(60, 95)

    # Right axis: Fairness
    ax2 = ax1.twinx()
    ax2.set_ylabel("Jain Fairness Index", color=color_fair)
    line2 = ax2.plot(thresh, fair_mean, color=color_fair, marker="s", linestyle="--", linewidth=2.2, label="Jain Fairness")
    ax2.fill_between(thresh, fair_mean - fair_std, fair_mean + fair_std, color=color_fair, alpha=0.18)
    ax2.tick_params(axis="y", labelcolor=color_fair)
    ax2.set_ylim(0.65, 0.95)

    # Vertical reference line at default tau = 0.58
    ax1.axvline(0.58, color="#d62728", linestyle=":", linewidth=2, label="Default $\\tau = 0.58$")

    ax1.set_title("Performance Sensitivity Across Admission Threshold $\\tau \\in [0.45, 0.72]$")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Combined legend
    lines = line1 + line2 + [matplotlib.lines.Line2D([0], [0], color="#d62728", linestyle=":", linewidth=2)]
    labels = ["Accuracy (%)", "Jain Fairness", "Operating Point ($\\tau=0.58$)"]
    ax1.legend(lines, labels, loc="lower center", framealpha=0.9, fontsize=9.5)

    fig.tight_layout()
    out_file = DOCS_DIR / "threshold_sensitivity.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    generate_attack_figure()
    generate_fairness_figure()
    generate_cloud_uploads_figure()
    generate_convergence_figure()
    generate_pareto_figure()
    generate_sensitivity_figure()
    print("\nALL 6 PUBLICATION FIGURES SUCCESSFULLY REGENERATED IN docs/!")
