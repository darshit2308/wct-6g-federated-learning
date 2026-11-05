import pandas as pd
from pathlib import Path

def generate_tables():
    clean_csv = Path("results/summary.csv")
    attack_csv = Path("results_attack/combined_attack_summary.csv")
    
    clean_df = pd.read_csv(clean_csv)
    attack_df = pd.read_csv(attack_csv)

    # 1. Clean Benchmark Table
    clean_tex = [
        "\\begin{table*}[t]",
        "\\caption{Performance Comparison on the Clean Benchmark (5 Independent Seeds)}",
        "\\label{tab:results_clean}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Final Accuracy} & \\textbf{Macro-F1} & \\textbf{Balanced Acc.} & \\textbf{Worst Class Acc.} & \\textbf{Jain Fairness} & \\textbf{Latency (ms)} \\\\",
        "\\midrule"
    ]
    
    for _, row in clean_df.iterrows():
        method_name = row['method'].replace('_', '\\_')
        acc = f"${row['final_accuracy_mean']:.4f} \\pm {row['final_accuracy_std']:.4f}$"
        f1 = f"${row['final_macro_f1_mean']:.4f} \\pm {row['final_macro_f1_std']:.4f}$"
        bal = f"${row['final_balanced_accuracy_mean']:.4f} \\pm {row['final_balanced_accuracy_std']:.4f}$"
        worst = f"${row['final_worst_class_accuracy_mean']:.4f} \\pm {row['final_worst_class_accuracy_std']:.4f}$"
        fair = f"${row['final_selection_fairness_mean']:.4f} \\pm {row['final_selection_fairness_std']:.4f}$"
        lat = f"${row['mean_round_latency_proxy_ms_mean']:.1f}$"
        clean_tex.append(f"\\texttt{{{method_name}}} & {acc} & {f1} & {bal} & {worst} & {fair} & {lat} \\\\")
        
    clean_tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        ""
    ])
    
    # 2. Attack Benchmark Table
    attack_tex = [
        "\\begin{table*}[t]",
        "\\caption{Adversarial Robustness Under Diverse Poisoning Attacks (25\\% Attackers, 5 Independent Seeds)}",
        "\\label{tab:results_attack}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Attack Model} & \\textbf{Method} & \\textbf{Final Accuracy} & \\textbf{Macro-F1} & \\textbf{Balanced Acc.} & \\textbf{Jain Fairness} & \\textbf{Detection Rate} \\\\",
        "\\midrule"
    ]
    
    for attack_type, group in attack_df.groupby("attack_type"):
        attack_label = attack_type.replace('_', ' ').title()
        attack_tex.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{attack_label} Attack (25\\% fraction)}}}} \\\\")
        for _, row in group.iterrows():
            method_name = row['method'].replace('_', '\\_')
            acc = f"${row['final_accuracy_mean']:.4f} \\pm {row['final_accuracy_std']:.4f}$"
            f1 = f"${row['final_macro_f1_mean']:.4f} \\pm {row['final_macro_f1_std']:.4f}$"
            bal = f"${row['final_balanced_accuracy_mean']:.4f} \\pm {row['final_balanced_accuracy_std']:.4f}$"
            fair = f"${row['final_selection_fairness_mean']:.4f} \\pm {row['final_selection_fairness_std']:.4f}$"
            det = f"${row['attack_detection_rate_mean']:.4f} \\pm {row['attack_detection_rate_std']:.4f}$"
            attack_tex.append(f" & \\texttt{{{method_name}}} & {acc} & {f1} & {bal} & {fair} & {det} \\\\")
        attack_tex.append("\\midrule")
        
    # Remove trailing \midrule and close table
    if attack_tex[-1] == "\\midrule":
        attack_tex.pop()
    attack_tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        ""
    ])
    
    Path("results/table_clean_publication.tex").write_text("\n".join(clean_tex), encoding="utf-8")
    Path("results/table_attack_publication.tex").write_text("\n".join(attack_tex), encoding="utf-8")
    Path("docs/table_clean_publication.tex").write_text("\n".join(clean_tex), encoding="utf-8")
    Path("docs/table_attack_publication.tex").write_text("\n".join(attack_tex), encoding="utf-8")
    print("Exported publication tables to results/ and docs/")

if __name__ == "__main__":
    generate_tables()
