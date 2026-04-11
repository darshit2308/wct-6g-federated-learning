import numpy as np

from model import apply_weight_delta


class SmartAggregator:
    def __init__(self, mad_threshold=3.5, min_updates_for_filtering=4):
        self.mad_threshold = mad_threshold
        self.min_updates_for_filtering = min_updates_for_filtering

    def detect_outliers(self, client_updates):
        """
        Remove suspicious client updates using a robust MAD-based norm filter.
        """
        if not client_updates:
            return []

        if len(client_updates) < self.min_updates_for_filtering:
            print(
                "   [Security] Skipping outlier filtering because the edge received "
                "too few client updates for a reliable robust estimate."
            )
            return client_updates

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
            print(
                "   [Security] Skipping outlier filtering because update norms are "
                "too similar for robust separation."
            )
            return client_updates

        safe_updates = []
        for update, norm in zip(client_updates, delta_norms):
            robust_z_score = abs(norm - median_norm) / (1.4826 * mad)
            if robust_z_score <= self.mad_threshold:
                safe_updates.append(update)
            else:
                print(
                    f"   [Security] Removed outlier update from {update['client_id']} "
                    f"(delta_norm={norm:.4f})"
                )

        return safe_updates

    def aggregate(self, safe_updates, base_weights):
        """
        Aggregate client deltas and apply the result to the incoming base model.
        """
        if not safe_updates:
            print("No valid client updates to aggregate at the edge.")
            return None

        print(f"Aggregating {len(safe_updates)} safe client updates...")

        total_weight = sum(
            update["num_samples"] * update["quality_score"] for update in safe_updates
        )

        aggregated_delta = [np.zeros_like(layer) for layer in safe_updates[0]["delta"]]
        for update in safe_updates:
            client_weight = update["num_samples"] * update["quality_score"]
            ratio = client_weight / total_weight
            for layer_index, delta_layer in enumerate(update["delta"]):
                aggregated_delta[layer_index] += delta_layer * ratio

        aggregated_weights = apply_weight_delta(base_weights, aggregated_delta)

        return {
            "weights": aggregated_weights,
            "num_samples": sum(update["num_samples"] for update in safe_updates),
            "client_ids": [update["client_id"] for update in safe_updates],
            "avg_accuracy": float(np.mean([update["accuracy"] for update in safe_updates])),
            "avg_loss": float(np.mean([update["loss"] for update in safe_updates])),
        }
