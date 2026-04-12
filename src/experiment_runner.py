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
    fairness_temperature: float
    description: str


@dataclass
class ExperimentConfig:
    dataset_name: str = "synthetic"
    dataset_root: str = "data"
    download_dataset: bool = False
    num_clients: int = 20
    num_edges: int = 2
    clients_per_round: int = 4
    num_rounds: int = 8
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
    bytes_per_scalar: int = 4
    convergence_fraction_of_best: float = 0.9
    results_dir: str = "results"
    quiet: bool = False


DEFAULT_METHODS = {
    "proposed": MethodSpec(
        name="proposed",
        topology="hierarchical",
        selection_strategy="fairness_aware",
        aggregation_strategy="quality_weighted",
        use_outlier_filter=True,
        fairness_temperature=1.0,
        description="Full hierarchical method with fairness-aware selection, weighted aggregation, and anomaly filtering.",
    ),
    "flat_fedavg_random": MethodSpec(
        name="flat_fedavg_random",
        topology="flat",
        selection_strategy="random",
        aggregation_strategy="fedavg",
        use_outlier_filter=False,
        fairness_temperature=0.0,
        description="Traditional flat FedAvg with random client selection.",
    ),
    "hierarchical_random_fedavg": MethodSpec(
        name="hierarchical_random_fedavg",
        topology="hierarchical",
        selection_strategy="random",
        aggregation_strategy="fedavg",
        use_outlier_filter=False,
        fairness_temperature=0.0,
        description="Hierarchical aggregation without intelligent selection or edge security.",
    ),
    "hierarchical_intelligent_fedavg": MethodSpec(
        name="hierarchical_intelligent_fedavg",
        topology="hierarchical",
        selection_strategy="intelligent",
        aggregation_strategy="fedavg",
        use_outlier_filter=False,
        fairness_temperature=0.0,
        description="Intelligent selection with plain FedAvg aggregation.",
    ),
    "hierarchical_intelligent_no_filter": MethodSpec(
        name="hierarchical_intelligent_no_filter",
        topology="hierarchical",
        selection_strategy="fairness_aware",
        aggregation_strategy="quality_weighted",
        use_outlier_filter=False,
        fairness_temperature=1.0,
        description="Fairness-aware weighted hierarchy without anomaly filtering.",
    ),
    "hierarchical_intelligent_no_fairness": MethodSpec(
        name="hierarchical_intelligent_no_fairness",
        topology="hierarchical",
        selection_strategy="intelligent",
        aggregation_strategy="quality_weighted",
        use_outlier_filter=True,
        fairness_temperature=0.0,
        description="Weighted hierarchy with filtering but without fairness-aware selection.",
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
    safe_clients,
    client_uploads,
    cloud_uploads,
    payload_to_edge_scalars,
    payload_to_cloud_scalars,
    energy_proxy,
    mean_selected_latency_ms,
    max_selected_latency_ms,
    mean_train_time_proxy_ms,
    filtered_updates,
    selection_fairness,
    attack_clients,
    blocked_adversarial,
    accepted_adversarial,
    blocked_benign,
    security_recall,
    filter_precision,
    benign_retention,
    topology,
    backhaul_latency_ms,
    edge_compute_latency_ms,
    bytes_per_scalar,
):
    if topology == "hierarchical":
        round_latency_proxy_ms = max_selected_latency_ms + backhaul_latency_ms + edge_compute_latency_ms
    else:
        round_latency_proxy_ms = max_selected_latency_ms * 1.75

    communication_reduction = 0.0
    if client_uploads:
        communication_reduction = 1.0 - (cloud_uploads / client_uploads)

    payload_to_edge_bytes = payload_to_edge_scalars * bytes_per_scalar
    payload_to_cloud_bytes = payload_to_cloud_scalars * bytes_per_scalar

    return {
        "round": round_num,
        "accuracy": global_metrics["accuracy"],
        "loss": global_metrics["loss"],
        "selected_clients": selected_clients,
        "safe_clients": safe_clients,
        "client_uploads": client_uploads,
        "cloud_uploads": cloud_uploads,
        "payload_to_edge_scalars": payload_to_edge_scalars,
        "payload_to_cloud_scalars": payload_to_cloud_scalars,
        "payload_to_edge_bytes": payload_to_edge_bytes,
        "payload_to_cloud_bytes": payload_to_cloud_bytes,
        "total_payload_bytes": payload_to_edge_bytes + payload_to_cloud_bytes,
        "energy_proxy": energy_proxy,
        "mean_selected_latency_ms": mean_selected_latency_ms,
        "max_selected_latency_ms": max_selected_latency_ms,
        "mean_train_time_proxy_ms": mean_train_time_proxy_ms,
        "round_latency_proxy_ms": round_latency_proxy_ms,
        "filtered_updates": filtered_updates,
        "selection_fairness": selection_fairness,
        "attack_clients": attack_clients,
        "blocked_adversarial": blocked_adversarial,
        "accepted_adversarial": accepted_adversarial,
        "blocked_benign": blocked_benign,
        "security_recall": security_recall,
        "filter_precision": filter_precision,
        "benign_retention": benign_retention,
        "communication_reduction": communication_reduction,
        "safe_client_ratio": (safe_clients / selected_clients) if selected_clients else 0.0,
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


def _client_stats(clients, method_name, dataset_name, seed):
    rows = []
    for client in clients:
        rows.append(
            {
                "method": method_name,
                "dataset": dataset_name,
                "seed": seed,
                "client_id": client.client_id,
                "data_size": client.data_size,
                "selection_count": client.selection_count,
                "accepted_update_count": client.accepted_update_count,
                "rejected_update_count": client.rejected_update_count,
                "acceptance_rate": (
                    client.accepted_update_count / client.selection_count
                    if client.selection_count
                    else 0.0
                ),
                "selection_freshness_score": client.selection_freshness_score(),
                "diversity_score": client.data_diversity_score(),
                "historical_utility": client.historical_utility,
                "reliability_score": client.reliability_score,
                "battery_level": client.battery_level,
                "network_latency": client.network_latency,
            }
        )
    return rows


def _first_round_reaching_threshold(round_history, threshold):
    for record in round_history:
        if record["accuracy"] >= threshold:
            return record["round"]
    return len(round_history)


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
                fairness_temperature=method_spec.fairness_temperature,
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
            safe_clients = int(sum(update["num_safe"] for update in edge_updates))
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
            mean_train_time_proxy_ms = float(
                mean(update["mean_train_time_proxy_ms"] for update in edge_updates)
            )
            filtered_updates = int(sum(update["num_removed"] for update in edge_updates))
            attack_clients = int(sum(update["num_adversarial"] for update in edge_updates))
            blocked_adversarial = int(
                sum(update["num_blocked_adversarial"] for update in edge_updates)
            )
            accepted_adversarial = int(
                sum(update["num_accepted_adversarial"] for update in edge_updates)
            )
            blocked_benign = int(sum(update["num_blocked_benign"] for update in edge_updates))
            security_recall = blocked_adversarial / attack_clients if attack_clients else 0.0
            filter_precision = (
                blocked_adversarial / filtered_updates if filtered_updates else 0.0
            )
            benign_total = selected_clients - attack_clients
            benign_retention = (
                (benign_total - blocked_benign) / benign_total if benign_total > 0 else 1.0
            )
            selection_fairness = _jain_fairness(
                [client.selection_count for client in clients]
            )

            round_record = _build_round_record(
                round_num=round_num,
                global_metrics=global_metrics,
                selected_clients=selected_clients,
                safe_clients=safe_clients,
                client_uploads=client_uploads,
                cloud_uploads=cloud_uploads,
                payload_to_edge_scalars=payload_to_edge_scalars,
                payload_to_cloud_scalars=payload_to_cloud_scalars,
                energy_proxy=energy_proxy,
                mean_selected_latency_ms=mean_selected_latency_ms,
                max_selected_latency_ms=max_selected_latency_ms,
                mean_train_time_proxy_ms=mean_train_time_proxy_ms,
                filtered_updates=filtered_updates,
                selection_fairness=selection_fairness,
                attack_clients=attack_clients,
                blocked_adversarial=blocked_adversarial,
                accepted_adversarial=accepted_adversarial,
                blocked_benign=blocked_benign,
                security_recall=security_recall,
                filter_precision=filter_precision,
                benign_retention=benign_retention,
                topology=method_spec.topology,
                backhaul_latency_ms=config.backhaul_latency_ms,
                edge_compute_latency_ms=config.edge_compute_latency_ms,
                bytes_per_scalar=config.bytes_per_scalar,
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
            fairness_temperature=method_spec.fairness_temperature,
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
            for client in selected_clients:
                client.mark_selected()
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
                accepted_ids = set(aggregation_summary["client_ids"])
            else:
                accepted_ids = set()

            for client in selected_clients:
                client_update = next(
                    update for update in client_updates if update["client_id"] == client.client_id
                )
                client.update_after_round(client_update, accepted=client.client_id in accepted_ids)

            cloud.global_weights = global_weights
            global_metrics = cloud.evaluate_global_model(evaluation_x, evaluation_y)

            selected_count = len(selected_clients)
            safe_count = aggregation_summary["num_safe"] if aggregation_summary is not None else 0
            payload_to_cloud_scalars = int(sum(update["upload_scalars"] for update in client_updates))
            mean_selected_latency_ms = float(
                np.mean([update["latency_ms"] for update in client_updates])
            )
            max_selected_latency_ms = float(
                np.max([update["latency_ms"] for update in client_updates])
            )
            mean_train_time_proxy_ms = float(
                np.mean([update["train_time_proxy_ms"] for update in client_updates])
            )
            energy_proxy = float(sum(update["energy_proxy"] for update in client_updates))
            filtered_updates = (
                aggregation_summary["num_removed"] if aggregation_summary is not None else 0
            )
            attack_clients = int(
                sum(1 for update in client_updates if update["is_adversarial"])
            )
            blocked_adversarial = (
                aggregation_summary["num_blocked_adversarial"] if aggregation_summary is not None else 0
            )
            accepted_adversarial = (
                aggregation_summary["num_accepted_adversarial"] if aggregation_summary is not None else 0
            )
            blocked_benign = (
                aggregation_summary["num_blocked_benign"] if aggregation_summary is not None else 0
            )
            security_recall = blocked_adversarial / attack_clients if attack_clients else 0.0
            filter_precision = (
                blocked_adversarial / filtered_updates if filtered_updates else 0.0
            )
            benign_total = selected_count - attack_clients
            benign_retention = (
                (benign_total - blocked_benign) / benign_total if benign_total > 0 else 1.0
            )
            selection_fairness = _jain_fairness(
                [client.selection_count for client in clients]
            )

            round_record = _build_round_record(
                round_num=round_num,
                global_metrics=global_metrics,
                selected_clients=selected_count,
                safe_clients=safe_count,
                client_uploads=selected_count,
                cloud_uploads=selected_count,
                payload_to_edge_scalars=0,
                payload_to_cloud_scalars=payload_to_cloud_scalars,
                energy_proxy=energy_proxy,
                mean_selected_latency_ms=mean_selected_latency_ms,
                max_selected_latency_ms=max_selected_latency_ms,
                mean_train_time_proxy_ms=mean_train_time_proxy_ms,
                filtered_updates=filtered_updates,
                selection_fairness=selection_fairness,
                attack_clients=attack_clients,
                blocked_adversarial=blocked_adversarial,
                accepted_adversarial=accepted_adversarial,
                blocked_benign=blocked_benign,
                security_recall=security_recall,
                filter_precision=filter_precision,
                benign_retention=benign_retention,
                topology=method_spec.topology,
                backhaul_latency_ms=config.backhaul_latency_ms,
                edge_compute_latency_ms=config.edge_compute_latency_ms,
                bytes_per_scalar=config.bytes_per_scalar,
            )
            round_history.append(round_record)

    final_round = round_history[-1]
    selection_counts = [client.selection_count for client in clients]
    best_accuracy = max(record["accuracy"] for record in round_history)
    convergence_threshold = config.convergence_fraction_of_best * best_accuracy
    total_attack_clients = int(sum(record["attack_clients"] for record in round_history))
    total_blocked_adversarial = int(sum(record["blocked_adversarial"] for record in round_history))
    total_blocked_benign = int(sum(record["blocked_benign"] for record in round_history))
    total_benign_selections = int(
        sum(record["selected_clients"] - record["attack_clients"] for record in round_history)
    )
    summary = {
        "method": method_spec.name,
        "description": method_spec.description,
        "dataset": bundle.dataset_name,
        "seed": config.seed,
        "num_clients": config.num_clients,
        "num_edges": config.num_edges,
        "clients_per_round": config.clients_per_round,
        "num_rounds": config.num_rounds,
        "global_model_scalars": global_model_size,
        "final_accuracy": final_round["accuracy"],
        "final_loss": final_round["loss"],
        "best_accuracy": best_accuracy,
        "accuracy_gain": final_round["accuracy"] - round_history[0]["accuracy"],
        "loss_drop": round_history[0]["loss"] - final_round["loss"],
        "accuracy_auc": float(mean(record["accuracy"] for record in round_history)),
        "round_to_convergence": _first_round_reaching_threshold(round_history, convergence_threshold),
        "total_client_uploads": int(sum(record["client_uploads"] for record in round_history)),
        "total_cloud_uploads": int(sum(record["cloud_uploads"] for record in round_history)),
        "total_payload_to_edge_scalars": int(
            sum(record["payload_to_edge_scalars"] for record in round_history)
        ),
        "total_payload_to_cloud_scalars": int(
            sum(record["payload_to_cloud_scalars"] for record in round_history)
        ),
        "total_payload_to_edge_bytes": int(
            sum(record["payload_to_edge_bytes"] for record in round_history)
        ),
        "total_payload_to_cloud_bytes": int(
            sum(record["payload_to_cloud_bytes"] for record in round_history)
        ),
        "total_payload_bytes": int(sum(record["total_payload_bytes"] for record in round_history)),
        "mean_round_latency_proxy_ms": float(
            mean(record["round_latency_proxy_ms"] for record in round_history)
        ),
        "mean_train_time_proxy_ms": float(
            mean(record["mean_train_time_proxy_ms"] for record in round_history)
        ),
        "total_energy_proxy": float(sum(record["energy_proxy"] for record in round_history)),
        "mean_communication_reduction": float(
            mean(record["communication_reduction"] for record in round_history)
        ),
        "final_selection_fairness": _jain_fairness(selection_counts),
        "client_coverage_ratio": float(
            sum(1 for count in selection_counts if count > 0) / len(selection_counts)
        ),
        "max_selection_count": int(max(selection_counts)),
        "min_selection_count": int(min(selection_counts)),
        "selection_count_std": float(np.std(selection_counts)),
        "total_filtered_updates": int(sum(record["filtered_updates"] for record in round_history)),
        "total_attack_clients": total_attack_clients,
        "attack_detection_rate": (
            total_blocked_adversarial / total_attack_clients if total_attack_clients else 0.0
        ),
        "benign_retention_rate": (
            (total_benign_selections - total_blocked_benign) / total_benign_selections
            if total_benign_selections
            else 1.0
        ),
        "attack_fraction": config.attack_fraction,
    }

    return {
        "method": method_spec.name,
        "summary": summary,
        "round_history": round_history,
        "client_stats": _client_stats(clients, method_spec.name, bundle.dataset_name, config.seed),
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
            "safe_clients",
            "client_uploads",
            "cloud_uploads",
            "payload_to_edge_scalars",
            "payload_to_cloud_scalars",
            "payload_to_edge_bytes",
            "payload_to_cloud_bytes",
            "total_payload_bytes",
            "energy_proxy",
            "mean_selected_latency_ms",
            "max_selected_latency_ms",
            "mean_train_time_proxy_ms",
            "round_latency_proxy_ms",
            "filtered_updates",
            "selection_fairness",
            "attack_clients",
            "blocked_adversarial",
            "accepted_adversarial",
            "blocked_benign",
            "security_recall",
            "filter_precision",
            "benign_retention",
            "communication_reduction",
            "safe_client_ratio",
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
            "description": method_summaries[0]["description"],
            "num_runs": len(method_summaries),
        }
        numeric_fields = [
            "final_accuracy",
            "final_loss",
            "best_accuracy",
            "accuracy_gain",
            "loss_drop",
            "accuracy_auc",
            "round_to_convergence",
            "total_client_uploads",
            "total_cloud_uploads",
            "total_payload_to_edge_scalars",
            "total_payload_to_cloud_scalars",
            "total_payload_to_edge_bytes",
            "total_payload_to_cloud_bytes",
            "total_payload_bytes",
            "mean_round_latency_proxy_ms",
            "mean_train_time_proxy_ms",
            "total_energy_proxy",
            "mean_communication_reduction",
            "final_selection_fairness",
            "client_coverage_ratio",
            "max_selection_count",
            "min_selection_count",
            "selection_count_std",
            "total_filtered_updates",
            "total_attack_clients",
            "attack_detection_rate",
            "benign_retention_rate",
        ]
        for field in numeric_fields:
            values = [summary[field] for summary in method_summaries]
            aggregated[f"{field}_mean"] = mean(values)
            aggregated[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregated_summaries.append(aggregated)
    return aggregated_summaries


def _aggregate_client_stats(client_rows):
    grouped = {}
    for row in client_rows:
        key = (row["method"], row["client_id"])
        grouped.setdefault(key, []).append(row)

    aggregated_rows = []
    numeric_fields = [
        "data_size",
        "selection_count",
        "accepted_update_count",
        "rejected_update_count",
        "acceptance_rate",
        "selection_freshness_score",
        "diversity_score",
        "historical_utility",
        "reliability_score",
        "battery_level",
        "network_latency",
    ]
    for (method, client_id), rows in sorted(grouped.items()):
        aggregated = {
            "method": method,
            "client_id": client_id,
            "num_runs": len(rows),
        }
        for field in numeric_fields:
            values = [row[field] for row in rows]
            aggregated[f"{field}_mean"] = mean(values)
            aggregated[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregated_rows.append(aggregated)
    return aggregated_rows


def _build_publication_report(aggregated_summaries):
    if not aggregated_summaries:
        return "# Publication Benchmark Report\n\nNo results were exported.\n"

    ranked = sorted(aggregated_summaries, key=lambda item: item["final_accuracy_mean"], reverse=True)
    best_accuracy = ranked[0]
    lowest_latency = min(aggregated_summaries, key=lambda item: item["mean_round_latency_proxy_ms_mean"])
    best_fairness = max(aggregated_summaries, key=lambda item: item["final_selection_fairness_mean"])
    strongest_security = max(aggregated_summaries, key=lambda item: item["attack_detection_rate_mean"])

    lines = [
        "# Publication Benchmark Report",
        "",
        "## Headline Findings",
        "",
        (
            f"- Best final accuracy: `{best_accuracy['method']}` at "
            f"{best_accuracy['final_accuracy_mean']:.4f} +- {best_accuracy['final_accuracy_std']:.4f}"
        ),
        (
            f"- Lowest latency proxy: `{lowest_latency['method']}` at "
            f"{lowest_latency['mean_round_latency_proxy_ms_mean']:.2f} ms"
        ),
        (
            f"- Best fairness: `{best_fairness['method']}` at "
            f"{best_fairness['final_selection_fairness_mean']:.4f}"
        ),
        (
            f"- Best attack detection: `{strongest_security['method']}` at "
            f"{strongest_security['attack_detection_rate_mean']:.4f}"
        ),
        "",
        "## Method Rankings",
        "",
    ]

    for rank, summary in enumerate(ranked, start=1):
        lines.append(
            (
                f"{rank}. `{summary['method']}`: accuracy={summary['final_accuracy_mean']:.4f}, "
                f"gain={summary['accuracy_gain_mean']:.4f}, "
                f"bytes={summary['total_payload_bytes_mean']:.0f}, "
                f"fairness={summary['final_selection_fairness_mean']:.4f}, "
                f"convergence_round={summary['round_to_convergence_mean']:.2f}"
            )
        )

    lines.extend(
        [
            "",
            "## Ablation Guide",
            "",
            "- `flat_fedavg_random` is the classic flat baseline.",
            "- `hierarchical_random_fedavg` removes intelligent client selection.",
            "- `hierarchical_intelligent_fedavg` removes quality-weighted aggregation.",
            "- `hierarchical_intelligent_no_filter` removes anomaly filtering.",
            "- `hierarchical_intelligent_no_fairness` removes fairness-aware participation balancing.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_results(results_dir, config, summaries, round_records, client_rows):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.csv"
    round_path = results_dir / "round_metrics.csv"
    client_path = results_dir / "client_participation.csv"
    report_path = results_dir / "publication_report.md"
    config_path = results_dir / "config.json"
    raw_path = results_dir / "raw_results.json"

    aggregated_summaries = _aggregate_summaries(summaries)
    aggregated_round_records = _aggregate_round_records(round_records)
    aggregated_client_rows = _aggregate_client_stats(client_rows)

    _write_csv(summary_path, aggregated_summaries)
    _write_csv(round_path, aggregated_round_records)
    _write_csv(client_path, aggregated_client_rows)
    report_path.write_text(_build_publication_report(aggregated_summaries), encoding="utf-8")

    config_payload = asdict(config)
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    raw_path.write_text(
        json.dumps(
            {
                "summaries": summaries,
                "round_records": round_records,
                "client_rows": client_rows,
                "config": config_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "summary_csv": str(summary_path),
        "round_csv": str(round_path),
        "client_csv": str(client_path),
        "report_md": str(report_path),
        "config_json": str(config_path),
        "raw_json": str(raw_path),
    }


def run_benchmark_suite(config, method_names, seeds):
    summaries = []
    round_records = []
    client_rows = []
    for method_name in method_names:
        method_spec = DEFAULT_METHODS[method_name]
        for seed in seeds:
            run_config = ExperimentConfig(**asdict(config))
            run_config.seed = seed
            result = run_experiment(run_config, method_spec)
            summaries.append(result["summary"])
            client_rows.extend(result["client_stats"])
            for record in result["round_history"]:
                round_records.append({"method": method_name, "seed": seed, **record})

    export_paths = export_results(
        results_dir=config.results_dir,
        config=config,
        summaries=summaries,
        round_records=round_records,
        client_rows=client_rows,
    )
    aggregated_summaries = _aggregate_summaries(summaries)
    return {
        "summaries": summaries,
        "round_records": round_records,
        "client_rows": client_rows,
        "aggregated_summaries": aggregated_summaries,
        "export_paths": export_paths,
    }


def format_summary_table(aggregated_summaries):
    headers = [
        "Method",
        "Final Acc",
        "Acc Gain",
        "Bytes",
        "Latency",
        "Fairness",
        "Atk Detect",
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
                    f"{summary['accuracy_gain_mean']:.4f}",
                    f"{summary['total_payload_bytes_mean']:.0f}",
                    f"{summary['mean_round_latency_proxy_ms_mean']:.2f}",
                    f"{summary['final_selection_fairness_mean']:.4f}",
                    f"{summary['attack_detection_rate_mean']:.4f}",
                ]
            )
        )
    return "\n".join(lines)
