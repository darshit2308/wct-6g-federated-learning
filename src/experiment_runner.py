import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from cloud_server import CloudServer
from data_utils import build_dataset_bundle
from edge_server import EdgeServer
from model import count_model_scalars
from smart_aggregator import SmartAggregator


@dataclass(frozen=True)
class MethodSpec:
    name: str
    topology: str
    selection_strategy: str
    aggregation_strategy: str
    use_outlier_filter: bool
    description: str


@dataclass
class ExperimentConfig:
    dataset_name: str = "synthetic"
    dataset_root: str = "data"
    download_dataset: bool = False
    num_clients: int = 20
    num_edges: int = 2
    clients_per_round: int = 4
    num_rounds: int = 5
    local_epochs: int = 2
    learning_rate: float = 0.02
    hidden_size: int = 32
    seed: int = 42
    synthetic_feature_count: int = 10
    synthetic_min_samples: int = 240
    synthetic_max_samples: int = 900
    eval_ratio: float = 0.2
    dirichlet_alpha: float = 0.5
    attack_fraction: float = 0.0
    attack_type: str = "sign_flip"
    attack_scale: float = 5.0
    backhaul_latency_ms: float = 8.0
    edge_compute_latency_ms: float = 2.0
    results_dir: str = "results"
    quiet: bool = False


DEFAULT_METHODS = {
    "proposed": MethodSpec(
        name="proposed",
        topology="hierarchical",
        selection_strategy="intelligent",
        aggregation_strategy="quality_weighted",
        use_outlier_filter=True,
        description="Hierarchical FL with intelligent selection, weighted aggregation, and outlier filtering.",
    ),
    "flat_fedavg_random": MethodSpec(
        name="flat_fedavg_random",
        topology="flat",
        selection_strategy="random",
        aggregation_strategy="fedavg",
        use_outlier_filter=False,
        description="Traditional flat FedAvg with random client selection.",
    ),
    "hierarchical_random_fedavg": MethodSpec(
        name="hierarchical_random_fedavg",
        topology="hierarchical",
        selection_strategy="random",
        aggregation_strategy="fedavg",
        use_outlier_filter=False,
        description="Hierarchical aggregation without intelligent selection or edge security.",
    ),
    "hierarchical_intelligent_fedavg": MethodSpec(
        name="hierarchical_intelligent_fedavg",
        topology="hierarchical",
        selection_strategy="intelligent",
        aggregation_strategy="fedavg",
        use_outlier_filter=False,
        description="Intelligent selection with plain FedAvg aggregation.",
    ),
    "hierarchical_intelligent_no_filter": MethodSpec(
        name="hierarchical_intelligent_no_filter",
        topology="hierarchical",
        selection_strategy="intelligent",
        aggregation_strategy="quality_weighted",
        use_outlier_filter=False,
        description="Weighted intelligent hierarchy without anomaly filtering.",
    ),
}


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_clients_across_edges(clients, num_edges):
    group_size = len(clients) // num_edges
    client_groups = []
    start = 0
    for edge_index in range(num_edges):
        end = start + group_size
        if edge_index == num_edges - 1:
            end = len(clients)
        client_groups.append(clients[start:end])
        start = end
    return client_groups


def _jain_fairness(selection_counts):
    selection_counts = np.asarray(selection_counts, dtype=float)
    denominator = len(selection_counts) * np.sum(selection_counts**2)
    if denominator == 0.0:
        return 0.0
    return float((np.sum(selection_counts) ** 2) / denominator)


def _build_round_record(
    round_num,
    global_metrics,
    selected_clients,
    client_uploads,
    cloud_uploads,
    payload_to_edge_scalars,
    payload_to_cloud_scalars,
    energy_proxy,
    mean_selected_latency_ms,
    max_selected_latency_ms,
    filtered_updates,
    selection_fairness,
    attack_clients,
    topology,
    backhaul_latency_ms,
    edge_compute_latency_ms,
):
    if topology == "hierarchical":
        round_latency_proxy_ms = max_selected_latency_ms + backhaul_latency_ms + edge_compute_latency_ms
    else:
        round_latency_proxy_ms = max_selected_latency_ms * 1.75

    communication_reduction = 0.0
    if client_uploads:
        communication_reduction = 1.0 - (cloud_uploads / client_uploads)

    return {
        "round": round_num,
        "accuracy": global_metrics["accuracy"],
        "loss": global_metrics["loss"],
        "selected_clients": selected_clients,
        "client_uploads": client_uploads,
        "cloud_uploads": cloud_uploads,
        "payload_to_edge_scalars": payload_to_edge_scalars,
        "payload_to_cloud_scalars": payload_to_cloud_scalars,
        "energy_proxy": energy_proxy,
        "mean_selected_latency_ms": mean_selected_latency_ms,
        "max_selected_latency_ms": max_selected_latency_ms,
        "round_latency_proxy_ms": round_latency_proxy_ms,
        "filtered_updates": filtered_updates,
        "selection_fairness": selection_fairness,
        "attack_clients": attack_clients,
        "communication_reduction": communication_reduction,
    }


def _train_selected_clients(
    clients,
    global_weights,
    epochs,
    learning_rate,
    attack_fraction,
    attack_type,
    attack_scale,
    rng,
    verbose,
):
    if not clients:
        return []

    num_adversarial = int(round(len(clients) * attack_fraction))
    adversarial_ids = set()
    if num_adversarial > 0:
        adversarial_ids = set(
            client.client_id for client in rng.choice(clients, size=num_adversarial, replace=False)
        )

    client_updates = []
    for client in clients:
        client.selection_count += 1
        client_updates.append(
            client.train_local_model(
                global_weights,
                epochs=epochs,
                lr=learning_rate,
                attack_config={
                    "enabled": client.client_id in adversarial_ids,
                    "attack_type": attack_type,
                    "attack_scale": attack_scale,
                },
                verbose=verbose,
            )
        )
    return client_updates


def run_experiment(config, method_spec):
    _set_seed(config.seed)
    verbose = not config.quiet
    bundle = build_dataset_bundle(
        dataset_name=config.dataset_name,
        num_clients=config.num_clients,
        seed=config.seed,
        hidden_size=config.hidden_size,
        synthetic_feature_count=config.synthetic_feature_count,
        synthetic_min_samples=config.synthetic_min_samples,
        synthetic_max_samples=config.synthetic_max_samples,
        eval_ratio=config.eval_ratio,
        dataset_root=config.dataset_root,
        download_dataset=config.download_dataset,
        dirichlet_alpha=config.dirichlet_alpha,
    )

    clients = bundle.clients
    model_config = bundle.model_config
    evaluation_x = bundle.evaluation_x
    evaluation_y = bundle.evaluation_y
    global_model_size = count_model_scalars(CloudServer(model_config).global_weights)

    cloud = CloudServer(model_config)
    global_weights = cloud.global_weights
    round_history = []
    rng = np.random.default_rng(config.seed + 999)

    if method_spec.topology == "hierarchical":
        edges = [
            EdgeServer(
                edge_id=f"E{edge_index + 1}",
                model_config=model_config,
                random_state=config.seed + 101 + edge_index,
                selection_strategy=method_spec.selection_strategy,
                aggregation_strategy=method_spec.aggregation_strategy,
                use_outlier_filter=method_spec.use_outlier_filter,
                local_epochs=config.local_epochs,
                learning_rate=config.learning_rate,
            )
            for edge_index in range(config.num_edges)
        ]
        client_groups = _split_clients_across_edges(clients, config.num_edges)

        for round_num in range(1, config.num_rounds + 1):
            if verbose:
                print("\n==========================================")
                print(f"   {method_spec.name.upper()} | ROUND {round_num}")
                print("==========================================")

            edge_updates = []
            for edge, edge_clients in zip(edges, client_groups):
                summary = edge.process_round(
                    clients=edge_clients,
                    incoming_global_weights=global_weights,
                    required_clients=config.clients_per_round,
                    round_num=round_num,
                    attack_fraction=config.attack_fraction,
                    attack_type=config.attack_type,
                    attack_scale=config.attack_scale,
                    verbose=verbose,
                )
                if summary is not None:
                    edge_updates.append(summary)

            global_weights = cloud.hierarchical_aggregation(edge_updates, verbose=verbose)
            global_metrics = cloud.evaluate_global_model(evaluation_x, evaluation_y)

            selected_clients = int(sum(update["num_selected"] for update in edge_updates))
            client_uploads = int(sum(update["client_uploads"] for update in edge_updates))
            cloud_uploads = int(sum(update["cloud_uploads"] for update in edge_updates))
            payload_to_edge_scalars = int(
                sum(update["payload_to_edge_scalars"] for update in edge_updates)
            )
            payload_to_cloud_scalars = int(
                sum(update["payload_to_cloud_scalars"] for update in edge_updates)
            )
            energy_proxy = float(sum(update["energy_proxy"] for update in edge_updates))
            mean_selected_latency_ms = float(
                mean(update["mean_latency_ms"] for update in edge_updates)
            )
            max_selected_latency_ms = float(
                max(update["max_latency_ms"] for update in edge_updates)
            )
            filtered_updates = int(sum(update["num_removed"] for update in edge_updates))
            attack_clients = int(sum(update["num_adversarial"] for update in edge_updates))
            selection_fairness = _jain_fairness(
                [client.selection_count for client in clients]
            )

            round_record = _build_round_record(
                round_num=round_num,
                global_metrics=global_metrics,
                selected_clients=selected_clients,
                client_uploads=client_uploads,
                cloud_uploads=cloud_uploads,
                payload_to_edge_scalars=payload_to_edge_scalars,
                payload_to_cloud_scalars=payload_to_cloud_scalars,
                energy_proxy=energy_proxy,
                mean_selected_latency_ms=mean_selected_latency_ms,
                max_selected_latency_ms=max_selected_latency_ms,
                filtered_updates=filtered_updates,
                selection_fairness=selection_fairness,
                attack_clients=attack_clients,
                topology=method_spec.topology,
                backhaul_latency_ms=config.backhaul_latency_ms,
                edge_compute_latency_ms=config.edge_compute_latency_ms,
            )
            round_history.append(round_record)

    else:
        selector = EdgeServer(
            edge_id="FlatCoordinator",
            model_config=model_config,
            random_state=config.seed + 303,
            selection_strategy=method_spec.selection_strategy,
            aggregation_strategy=method_spec.aggregation_strategy,
            use_outlier_filter=method_spec.use_outlier_filter,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
        )
        aggregator = SmartAggregator()

        for round_num in range(1, config.num_rounds + 1):
            if verbose:
                print("\n==========================================")
                print(f"   {method_spec.name.upper()} | ROUND {round_num}")
                print("==========================================")

            selector.advance_round_state(clients)
            selected_clients = selector.select_clients(
                clients,
                required_clients=config.num_edges * config.clients_per_round,
                verbose=verbose,
            )
            client_updates = _train_selected_clients(
                clients=selected_clients,
                global_weights=global_weights,
                epochs=config.local_epochs,
                learning_rate=config.learning_rate,
                attack_fraction=config.attack_fraction,
                attack_type=config.attack_type,
                attack_scale=config.attack_scale,
                rng=rng,
                verbose=verbose,
            )

            aggregation_summary = aggregator.aggregate(
                client_updates,
                global_weights,
                aggregation_strategy=method_spec.aggregation_strategy,
                use_outlier_filter=method_spec.use_outlier_filter,
                verbose=verbose,
            )
            if aggregation_summary is not None:
                global_weights = aggregation_summary["weights"]
                cloud.global_weights = global_weights

            cloud.global_weights = global_weights
            global_metrics = cloud.evaluate_global_model(evaluation_x, evaluation_y)

            selected_count = len(selected_clients)
            payload_to_cloud_scalars = int(sum(update["upload_scalars"] for update in client_updates))
            mean_selected_latency_ms = float(
                np.mean([update["latency_ms"] for update in client_updates])
            )
            max_selected_latency_ms = float(
                np.max([update["latency_ms"] for update in client_updates])
            )
            energy_proxy = float(sum(update["energy_proxy"] for update in client_updates))
            filtered_updates = (
                aggregation_summary["num_removed"] if aggregation_summary is not None else 0
            )
            attack_clients = int(
                sum(1 for update in client_updates if update["is_adversarial"])
            )
            selection_fairness = _jain_fairness(
                [client.selection_count for client in clients]
            )

            round_record = _build_round_record(
                round_num=round_num,
                global_metrics=global_metrics,
                selected_clients=selected_count,
                client_uploads=selected_count,
                cloud_uploads=selected_count,
                payload_to_edge_scalars=0,
                payload_to_cloud_scalars=payload_to_cloud_scalars,
                energy_proxy=energy_proxy,
                mean_selected_latency_ms=mean_selected_latency_ms,
                max_selected_latency_ms=max_selected_latency_ms,
                filtered_updates=filtered_updates,
                selection_fairness=selection_fairness,
                attack_clients=attack_clients,
                topology=method_spec.topology,
                backhaul_latency_ms=config.backhaul_latency_ms,
                edge_compute_latency_ms=config.edge_compute_latency_ms,
            )
            round_history.append(round_record)

    final_round = round_history[-1]
    selection_counts = [client.selection_count for client in clients]
    summary = {
        "method": method_spec.name,
        "dataset": bundle.dataset_name,
        "seed": config.seed,
        "num_clients": config.num_clients,
        "num_edges": config.num_edges,
        "clients_per_round": config.clients_per_round,
        "num_rounds": config.num_rounds,
        "global_model_scalars": global_model_size,
        "final_accuracy": final_round["accuracy"],
        "final_loss": final_round["loss"],
        "best_accuracy": max(record["accuracy"] for record in round_history),
        "accuracy_gain": final_round["accuracy"] - round_history[0]["accuracy"],
        "loss_drop": round_history[0]["loss"] - final_round["loss"],
        "total_client_uploads": int(sum(record["client_uploads"] for record in round_history)),
        "total_cloud_uploads": int(sum(record["cloud_uploads"] for record in round_history)),
        "total_payload_to_edge_scalars": int(
            sum(record["payload_to_edge_scalars"] for record in round_history)
        ),
        "total_payload_to_cloud_scalars": int(
            sum(record["payload_to_cloud_scalars"] for record in round_history)
        ),
        "mean_round_latency_proxy_ms": float(
            mean(record["round_latency_proxy_ms"] for record in round_history)
        ),
        "total_energy_proxy": float(sum(record["energy_proxy"] for record in round_history)),
        "final_selection_fairness": _jain_fairness(selection_counts),
        "client_coverage_ratio": float(
            sum(1 for count in selection_counts if count > 0) / len(selection_counts)
        ),
        "total_filtered_updates": int(sum(record["filtered_updates"] for record in round_history)),
        "attack_fraction": config.attack_fraction,
    }

    return {
        "method": method_spec.name,
        "summary": summary,
        "round_history": round_history,
    }


def _aggregate_round_records(round_records):
    grouped = {}
    for record in round_records:
        key = (record["method"], record["round"])
        grouped.setdefault(key, []).append(record)

    aggregated_records = []
    for (method, round_num), records in sorted(grouped.items()):
        aggregated_record = {
            "method": method,
            "round": round_num,
        }
        numeric_fields = [
            "accuracy",
            "loss",
            "selected_clients",
            "client_uploads",
            "cloud_uploads",
            "payload_to_edge_scalars",
            "payload_to_cloud_scalars",
            "energy_proxy",
            "mean_selected_latency_ms",
            "max_selected_latency_ms",
            "round_latency_proxy_ms",
            "filtered_updates",
            "selection_fairness",
            "attack_clients",
            "communication_reduction",
        ]
        for field in numeric_fields:
            values = [record[field] for record in records]
            aggregated_record[f"{field}_mean"] = mean(values)
            aggregated_record[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregated_records.append(aggregated_record)
    return aggregated_records


def _aggregate_summaries(summaries):
    grouped = {}
    for summary in summaries:
        grouped.setdefault(summary["method"], []).append(summary)

    aggregated_summaries = []
    for method, method_summaries in sorted(grouped.items()):
        aggregated = {
            "method": method,
            "dataset": method_summaries[0]["dataset"],
            "num_runs": len(method_summaries),
        }
        numeric_fields = [
            "final_accuracy",
            "final_loss",
            "best_accuracy",
            "accuracy_gain",
            "loss_drop",
            "total_client_uploads",
            "total_cloud_uploads",
            "total_payload_to_edge_scalars",
            "total_payload_to_cloud_scalars",
            "mean_round_latency_proxy_ms",
            "total_energy_proxy",
            "final_selection_fairness",
            "client_coverage_ratio",
            "total_filtered_updates",
        ]
        for field in numeric_fields:
            values = [summary[field] for summary in method_summaries]
            aggregated[f"{field}_mean"] = mean(values)
            aggregated[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregated_summaries.append(aggregated)
    return aggregated_summaries


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_results(results_dir, config, summaries, round_records):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.csv"
    round_path = results_dir / "round_metrics.csv"
    config_path = results_dir / "config.json"
    raw_path = results_dir / "raw_results.json"

    aggregated_summaries = _aggregate_summaries(summaries)
    aggregated_round_records = _aggregate_round_records(round_records)

    _write_csv(summary_path, aggregated_summaries)
    _write_csv(round_path, aggregated_round_records)

    config_payload = asdict(config)
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    raw_path.write_text(
        json.dumps(
            {
                "summaries": summaries,
                "round_records": round_records,
                "config": config_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "summary_csv": str(summary_path),
        "round_csv": str(round_path),
        "config_json": str(config_path),
        "raw_json": str(raw_path),
    }


def run_benchmark_suite(config, method_names, seeds):
    summaries = []
    round_records = []
    for method_name in method_names:
        method_spec = DEFAULT_METHODS[method_name]
        for seed in seeds:
            run_config = ExperimentConfig(**asdict(config))
            run_config.seed = seed
            result = run_experiment(run_config, method_spec)
            summaries.append(result["summary"])
            for record in result["round_history"]:
                round_records.append({"method": method_name, "seed": seed, **record})

    export_paths = export_results(
        results_dir=config.results_dir,
        config=config,
        summaries=summaries,
        round_records=round_records,
    )
    aggregated_summaries = _aggregate_summaries(summaries)
    return {
        "summaries": summaries,
        "round_records": round_records,
        "aggregated_summaries": aggregated_summaries,
        "export_paths": export_paths,
    }


def format_summary_table(aggregated_summaries):
    headers = [
        "Method",
        "Final Acc",
        "Final Loss",
        "Acc Gain",
        "Cloud Uploads",
        "Energy Proxy",
        "Fairness",
    ]
    lines = [
        " | ".join(headers),
        " | ".join(["---"] * len(headers)),
    ]

    for summary in aggregated_summaries:
        lines.append(
            " | ".join(
                [
                    summary["method"],
                    f"{summary['final_accuracy_mean']:.4f}",
                    f"{summary['final_loss_mean']:.4f}",
                    f"{summary['accuracy_gain_mean']:.4f}",
                    f"{summary['total_cloud_uploads_mean']:.1f}",
                    f"{summary['total_energy_proxy_mean']:.2f}",
                    f"{summary['final_selection_fairness_mean']:.4f}",
                ]
            )
        )
    return "\n".join(lines)
