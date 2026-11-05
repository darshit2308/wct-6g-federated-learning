import argparse
import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from experiment_runner import ExperimentConfig, run_benchmark_suite, format_summary_table


def run_clean_suite(dataset="digits", rounds=10, seeds=[42, 52, 62, 72, 82]):
    print("\n" + "=" * 60)
    print(">>> 1. RUNNING FULL CLEAN BENCHMARK SUITE (5 SEEDS) <<<")
    print("=" * 60)

    clean_methods = [
        "proposed",
        "flat_fedavg_random",
        "hierarchical_random_fedavg",
        "hierarchical_intelligent_fedavg",
        "hierarchical_intelligent_no_filter",
        "hierarchical_intelligent_no_fairness",
        "hierarchical_krum",
        "hierarchical_trimmed_mean",
        "hierarchical_direct_score",
        "hierarchical_fairness_direct_score",
    ]

    config = ExperimentConfig(
        dataset_name=dataset,
        num_clients=20,
        num_edges=2,
        clients_per_round=4,
        num_rounds=rounds,
        local_epochs=4,
        learning_rate=0.01,
        hidden_size=64,
        results_dir="results",
        attack_fraction=0.0,
        quiet=True,
    )

    suite_res = run_benchmark_suite(config, method_names=clean_methods, seeds=seeds)
    print("\n--- CLEAN BENCHMARK SUMMARY ---")
    print(format_summary_table(suite_res["aggregated_summaries"]))
    print("\nExported files in results/:", suite_res["export_paths"])
    return suite_res


def run_adversarial_suite(dataset="digits", rounds=10, seeds=[42, 52, 62, 72, 82]):
    print("\n" + "=" * 60)
    print(">>> 2. RUNNING ADVERSARIAL BENCHMARK SUITE (SIGN-FLIP & ADAPTIVE FANG) <<<")
    print("=" * 60)

    attack_methods = [
        "proposed",
        "hierarchical_intelligent_no_filter",
        "hierarchical_krum",
        "hierarchical_trimmed_mean",
        "flat_fedavg_random",
    ]

    attacks = [
        ("sign_flip", 0.25, 5.0, "results_attack_signflip"),
        ("adaptive_fang", 0.25, 1.0, "results_attack_fang"),
        ("label_flip", 0.25, 1.0, "results_attack_labelflip"),
    ]

    all_attack_summaries = []

    for attack_type, frac, scale, out_dir in attacks:
        print(f"\n--- Running Attack Benchmark: {attack_type} (fraction={frac}, scale={scale}) ---")
        config = ExperimentConfig(
            dataset_name=dataset,
            num_clients=20,
            num_edges=2,
            clients_per_round=4,
            num_rounds=rounds,
            local_epochs=4,
            learning_rate=0.01,
            hidden_size=64,
            results_dir=out_dir,
            attack_fraction=frac,
            attack_type=attack_type,
            attack_scale=scale,
            quiet=True,
        )
        res = run_benchmark_suite(config, method_names=attack_methods, seeds=seeds)
        print(f"\nSummary for {attack_type}:")
        print(format_summary_table(res["aggregated_summaries"]))

        for s in res["aggregated_summaries"]:
            s_copy = dict(s)
            s_copy["attack_type"] = attack_type
            s_copy["attack_fraction"] = frac
            all_attack_summaries.append(s_copy)

    # Export combined attack summary
    out_path = Path("results_attack")
    out_path.mkdir(parents=True, exist_ok=True)
    combined_df = pd.DataFrame(all_attack_summaries)
    combined_csv = out_path / "combined_attack_summary.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nCombined attack summary saved to {combined_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seeds", default="42,52,62,72,82")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-attacks", action="store_true")
    args = parser.parse_args()

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if not args.skip_clean:
        run_clean_suite(rounds=args.rounds, seeds=seed_list)

    if not args.skip_attacks:
        run_adversarial_suite(rounds=args.rounds, seeds=seed_list)


if __name__ == "__main__":
    main()
