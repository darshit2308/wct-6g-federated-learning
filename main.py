import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from experiment_runner import (  # noqa: E402
    DEFAULT_METHODS,
    ExperimentConfig,
    format_summary_table,
    run_benchmark_suite,
    run_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Research-grade benchmark runner for hierarchical FL over 6G-style edge networks."
    )
    parser.add_argument(
        "--mode",
        choices=["suite", "single"],
        default="suite",
        help="Run the full benchmark suite or a single method.",
    )
    parser.add_argument(
        "--method",
        default="proposed",
        choices=sorted(DEFAULT_METHODS.keys()),
        help="Method to use in single-run mode.",
    )
    parser.add_argument(
        "--methods",
        default="proposed,flat_fedavg_random,hierarchical_random_fedavg,hierarchical_intelligent_fedavg,hierarchical_intelligent_no_filter",
        help="Comma-separated list of methods for suite mode.",
    )
    parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "digits", "mnist"],
        help="Dataset to use. digits is the strongest no-download benchmark. mnist requires torchvision data access.",
    )
    parser.add_argument("--download-dataset", action="store_true", help="Allow dataset downloads when supported.")
    parser.add_argument("--dataset-root", default="data", help="Directory for datasets.")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-edges", type=int, default=2)
    parser.add_argument("--clients-per-round", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument(
        "--seeds",
        default="42,52,62",
        help="Comma-separated seeds used in suite mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used in single-run mode.")
    parser.add_argument("--attack-fraction", type=float, default=0.0, help="Fraction of selected clients that behave adversarially.")
    parser.add_argument("--attack-type", choices=["sign_flip", "gaussian_noise"], default="sign_flip")
    parser.add_argument("--attack-scale", type=float, default=5.0)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-round logging.")
    return parser.parse_args()


def build_config(args, seed):
    return ExperimentConfig(
        dataset_name=args.dataset,
        dataset_root=args.dataset_root,
        download_dataset=args.download_dataset,
        num_clients=args.num_clients,
        num_edges=args.num_edges,
        clients_per_round=args.clients_per_round,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=seed,
        attack_fraction=args.attack_fraction,
        attack_type=args.attack_type,
        attack_scale=args.attack_scale,
        results_dir=args.results_dir,
        quiet=args.quiet,
    )


def main():
    args = parse_args()

    if args.mode == "single":
        config = build_config(args, seed=args.seed)
        result = run_experiment(config, DEFAULT_METHODS[args.method])
        summary = result["summary"]
        print("\n=== SINGLE RUN SUMMARY ===")
        print(f"Method: {summary['method']}")
        print(f"Dataset: {summary['dataset']}")
        print(f"Final Accuracy: {summary['final_accuracy']:.4f}")
        print(f"Final Loss: {summary['final_loss']:.4f}")
        print(f"Accuracy Gain: {summary['accuracy_gain']:.4f}")
        print(f"Cloud Uploads: {summary['total_cloud_uploads']}")
        print(f"Energy Proxy: {summary['total_energy_proxy']:.2f}")
        print(f"Selection Fairness: {summary['final_selection_fairness']:.4f}")
        return

    method_names = [method.strip() for method in args.methods.split(",") if method.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    config = build_config(args, seed=seeds[0] if seeds else args.seed)
    suite_result = run_benchmark_suite(config, method_names=method_names, seeds=seeds)

    print("\n=== BENCHMARK SUMMARY ===")
    print(format_summary_table(suite_result["aggregated_summaries"]))
    print("\nExported files:")
    for label, path in suite_result["export_paths"].items():
        print(f" - {label}: {path}")


if __name__ == "__main__":
    main()
