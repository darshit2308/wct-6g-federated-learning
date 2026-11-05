import numpy as np

from model import apply_weight_delta


class SmartAggregator:
    def __init__(
        self,
        mad_threshold=3.5,
        min_updates_for_filtering=4,
        cosine_mad_threshold=3.0,
        cosine_similarity_floor=0.0,
    ):
        self.mad_threshold = mad_threshold
        self.min_updates_for_filtering = min_updates_for_filtering
        self.cosine_mad_threshold = cosine_mad_threshold
        self.cosine_similarity_floor = cosine_similarity_floor

    def _flatten_delta(self, update):
        return np.concatenate([layer.reshape(-1) for layer in update["delta"]]).astype(np.float64)

    def _attach_geometric_diagnostics(self, client_updates):
        if not client_updates:
            return

        flat_deltas = [self._flatten_delta(update) for update in client_updates]
        reference_delta = np.median(np.vstack(flat_deltas), axis=0)
        reference_norm = float(np.linalg.norm(reference_delta))

        for update, flat_delta in zip(client_updates, flat_deltas):
            delta_norm = float(np.linalg.norm(flat_delta))
            denominator = max(delta_norm * reference_norm, 1e-12)
            reference_cosine = float(np.dot(flat_delta, reference_delta) / denominator)
            update["delta_norm"] = delta_norm
            update["reference_cosine"] = reference_cosine

    def detect_outliers(self, client_updates, verbose=True):
        """
        Remove suspicious client updates using robust norm and direction filters.
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

        delta_norms = np.array([update["delta_norm"] for update in client_updates])
        cosine_scores = np.array([update.get("reference_cosine", 1.0) for update in client_updates])

        median_norm = float(np.median(delta_norms))
        mad = float(np.median(np.abs(delta_norms - median_norm)))
        cosine_median = float(np.median(cosine_scores))
        cosine_mad = float(np.median(np.abs(cosine_scores - cosine_median)))

        if mad == 0.0 and cosine_mad == 0.0:
            if verbose:
                print(
                    "   [Security] Skipping outlier filtering because update norms are "
                    "and directions are too similar for robust separation."
                )
            return client_updates, []

        safe_updates = []
        removed_updates = []
        for update, norm, cosine in zip(client_updates, delta_norms, cosine_scores):
            norm_flag = False
            cosine_flag = False

            if mad > 0.0:
                robust_z_score = abs(norm - median_norm) / (1.4826 * mad)
                norm_flag = robust_z_score > self.mad_threshold

            if cosine < self.cosine_similarity_floor:
                cosine_flag = True
            elif cosine_mad > 0.0 and cosine < cosine_median:
                cosine_z_score = abs(cosine - cosine_median) / (1.4826 * cosine_mad)
                cosine_flag = cosine_z_score > self.cosine_mad_threshold

            if not norm_flag and not cosine_flag:
                safe_updates.append(update)
            else:
                removed_updates.append(update)
                if verbose:
                    reasons = []
                    if norm_flag:
                        reasons.append(f"norm={norm:.4f}")
                    if cosine_flag:
                        reasons.append(f"cosine={cosine:.4f}")
                    print(
                        f"   [Security] Removed outlier update from {update['client_id']} "
                        f"({', '.join(reasons)})"
                    )

        return safe_updates, removed_updates

    def _unflatten_delta(self, flat_vector, template_delta):
        unflattened = []
        idx = 0
        for layer in template_delta:
            size = layer.size
            unflattened.append(
                flat_vector[idx : idx + size].reshape(layer.shape).astype(layer.dtype)
            )
            idx += size
        return unflattened

    def _krum_aggregate(self, client_updates, num_byzantine=None):
        """
        Krum aggregation rule (Blanchard et al., NeurIPS 2017).
        Selects the single update minimizing the sum of squared distances to its m - f - 2 closest neighbors.
        """
        m = len(client_updates)
        if m <= 2:
            return client_updates[0]["delta"]

        if num_byzantine is None:
            f = max(1, int(np.floor((m - 1) / 3)))
        else:
            f = num_byzantine

        k = max(1, min(m - f - 2, m - 1))
        flat_deltas = [self._flatten_delta(u) for u in client_updates]
        dist_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                d = float(np.sum((flat_deltas[i] - flat_deltas[j]) ** 2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        scores = []
        for i in range(m):
            dists = np.sort(np.delete(dist_matrix[i], i))
            scores.append(float(np.sum(dists[:k])))

        best_idx = int(np.argmin(scores))
        return client_updates[best_idx]["delta"]

    def _trimmed_mean_aggregate(self, client_updates, trim_ratio=0.15):
        """
        Coordinate-wise Trimmed Mean aggregation (Yin et al., ICML 2018).
        Removes the smallest and largest beta fraction along each coordinate.
        """
        m = len(client_updates)
        if m <= 2:
            return client_updates[0]["delta"]

        flat_deltas = np.vstack([self._flatten_delta(u) for u in client_updates])
        k = int(np.floor(m * trim_ratio))
        if 2 * k >= m:
            k = max(0, (m - 1) // 2)

        if k == 0:
            aggregated_flat = np.mean(flat_deltas, axis=0)
        else:
            sorted_deltas = np.sort(flat_deltas, axis=0)
            trimmed = sorted_deltas[k : m - k, :]
            aggregated_flat = np.mean(trimmed, axis=0)

        return self._unflatten_delta(aggregated_flat, client_updates[0]["delta"])

    def _client_weight(self, update, aggregation_strategy):
        if aggregation_strategy in ["uniform", "krum", "trimmed_mean"]:
            return 1.0
        if aggregation_strategy == "fedavg":
            return float(update["num_samples"])
        if aggregation_strategy == "quality_weighted":
            return float(update["num_samples"] * update["quality_score"])
        if aggregation_strategy == "adaptive_weighted":
            utility_term = 0.6 + 0.4 * update.get("utility_score", update["quality_score"])
            return float(update["num_samples"] * update["quality_score"] * utility_term)
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

        self._attach_geometric_diagnostics(client_updates)
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
            print(f"Aggregating {len(safe_updates)} safe client updates via {aggregation_strategy}...")

        if aggregation_strategy == "krum":
            aggregated_delta = self._krum_aggregate(safe_updates)
        elif aggregation_strategy == "trimmed_mean":
            aggregated_delta = self._trimmed_mean_aggregate(safe_updates)
        else:
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
        blocked_adversarial = int(sum(1 for update in removed_updates if update["is_adversarial"]))
        accepted_adversarial = int(sum(1 for update in safe_updates if update["is_adversarial"]))
        total_adversarial = blocked_adversarial + accepted_adversarial
        blocked_benign = int(sum(1 for update in removed_updates if not update["is_adversarial"]))
        accepted_benign = int(sum(1 for update in safe_updates if not update["is_adversarial"]))
        security_recall = (
            blocked_adversarial / total_adversarial if total_adversarial else 0.0
        )
        filter_precision = (
            blocked_adversarial / len(removed_updates) if removed_updates else 0.0
        )
        benign_retention = (
            accepted_benign / (accepted_benign + blocked_benign)
            if (accepted_benign + blocked_benign)
            else 1.0
        )

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
            "mean_train_time_proxy_ms": float(
                np.mean([update["train_time_proxy_ms"] for update in client_updates])
            ),
            "num_adversarial": int(
                sum(1 for update in client_updates if update["is_adversarial"])
            ),
            "num_blocked_adversarial": blocked_adversarial,
            "num_accepted_adversarial": accepted_adversarial,
            "num_blocked_benign": blocked_benign,
            "security_recall": float(security_recall),
            "filter_precision": float(filter_precision),
            "benign_retention": float(benign_retention),
            "mean_quality_score": float(
                np.mean([update["quality_score"] for update in safe_updates])
            ),
            "mean_reference_cosine": float(
                np.mean([update.get("reference_cosine", 1.0) for update in safe_updates])
            ),
            "min_reference_cosine": float(
                min(update.get("reference_cosine", 1.0) for update in safe_updates)
            ),
        }
