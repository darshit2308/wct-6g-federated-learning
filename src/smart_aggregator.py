import numpy as np

from model import apply_weight_delta


class SmartAggregator:
    def __init__(self, mad_threshold=3.5, min_updates_for_filtering=4):
        self.mad_threshold = mad_threshold
        self.min_updates_for_filtering = min_updates_for_filtering

    def detect_outliers(self, client_updates, verbose=True):
        """
        Remove suspicious client updates using a robust MAD-based norm filter.
        """
        if not client_updates:
            return client_updates, []

        if len(client_updates) < self.min_updates_for_filtering:
            if verbose:
                print(
                    "   [Security] Skipping outlier filtering because the edge received "
                    "too few updates for a reliable robust estimate."
                )
            return client_updates, []

        delta_norms = np.array(
            [
                np.linalg.norm(
                    np.concatenate([layer.reshape(-1) for layer in update["delta"]])
                )
                for update in client_updates
            ]
        )

        median_norm = float(np.median(delta_norms))
        mad = float(np.median(np.abs(delta_norms - median_norm)))

        if mad == 0.0:
            if verbose:
                print(
                    "   [Security] Skipping outlier filtering because update norms are "
                    "too similar for robust separation."
                )
            return client_updates, []

        safe_updates = []
        removed_updates = []
        for update, norm in zip(client_updates, delta_norms):
            robust_z_score = abs(norm - median_norm) / (1.4826 * mad)
            if robust_z_score <= self.mad_threshold:
                safe_updates.append(update)
            else:
                removed_updates.append(update)
                if verbose:
                    print(
                        f"   [Security] Removed outlier update from {update['client_id']} "
                        f"(delta_norm={norm:.4f})"
                    )

        return safe_updates, removed_updates

    def _client_weight(self, update, aggregation_strategy):
        if aggregation_strategy == "uniform":
            return 1.0
        if aggregation_strategy == "fedavg":
            return float(update["num_samples"])
        if aggregation_strategy == "quality_weighted":
            return float(update["num_samples"] * update["quality_score"])
        raise ValueError(f"Unsupported aggregation strategy: {aggregation_strategy}")

    def aggregate(
        self,
        client_updates,
        base_weights,
        aggregation_strategy="quality_weighted",
        use_outlier_filter=True,
        verbose=True,
    ):
        """
        Aggregate client deltas and apply the result to the incoming base model.
        """
        if not client_updates:
            if verbose:
                print("No valid client updates to aggregate.")
            return None

        removed_updates = []
        if use_outlier_filter:
            safe_updates, removed_updates = self.detect_outliers(
                client_updates, verbose=verbose
            )
        else:
            safe_updates = list(client_updates)

        if not safe_updates:
            if verbose:
                print("No safe updates remained after filtering.")
            return None

        if verbose:
            print(f"Aggregating {len(safe_updates)} safe client updates...")

        raw_weights = [
            self._client_weight(update, aggregation_strategy) for update in safe_updates
        ]
        total_weight = sum(raw_weights)
        normalized_weights = [weight / total_weight for weight in raw_weights]

        aggregated_delta = [np.zeros_like(layer) for layer in safe_updates[0]["delta"]]
        for ratio, update in zip(normalized_weights, safe_updates):
            for layer_index, delta_layer in enumerate(update["delta"]):
                aggregated_delta[layer_index] += delta_layer * ratio

        aggregated_weights = apply_weight_delta(base_weights, aggregated_delta)

        return {
            "weights": aggregated_weights,
            "num_samples": sum(update["num_samples"] for update in safe_updates),
            "client_ids": [update["client_id"] for update in safe_updates],
            "selected_client_ids": [update["client_id"] for update in client_updates],
            "removed_client_ids": [update["client_id"] for update in removed_updates],
            "avg_accuracy": float(np.mean([update["accuracy"] for update in safe_updates])),
            "avg_loss": float(np.mean([update["loss"] for update in safe_updates])),
            "num_removed": len(removed_updates),
            "num_selected": len(client_updates),
            "num_safe": len(safe_updates),
            "energy_proxy": float(sum(update["energy_proxy"] for update in client_updates)),
            "payload_scalars": int(sum(update["upload_scalars"] for update in client_updates)),
            "safe_payload_scalars": int(
                sum(update["upload_scalars"] for update in safe_updates)
            ),
            "mean_latency_ms": float(
                np.mean([update["latency_ms"] for update in client_updates])
            ),
            "max_latency_ms": float(max(update["latency_ms"] for update in client_updates)),
            "num_adversarial": int(
                sum(1 for update in client_updates if update["is_adversarial"])
            ),
        }
