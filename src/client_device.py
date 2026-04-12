import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import (
    apply_weight_delta,
    build_model,
    clone_weights,
    compute_weight_delta,
    get_model_weights,
    set_model_weights,
)


class DeviceClient:
    def __init__(
        self,
        client_id,
        battery_level,
        network_latency,
        train_x,
        train_y,
        eval_x,
        eval_y,
        seed,
        model_config,
        reliability_score=1.0,
    ):
        self.client_id = client_id
        self.battery_level = battery_level
        self.network_latency = network_latency
        self.seed = seed
        self.model_config = model_config
        self.reliability_score = reliability_score

        self.train_x = train_x.float()
        self.train_y = train_y.long()
        self.eval_x = eval_x.float()
        self.eval_y = eval_y.long()

        self.data_size = len(self.train_y)
        self.selection_count = 0
        self.accepted_update_count = 0
        self.rejected_update_count = 0
        self.rounds_since_last_selection = 3
        self.local_model = build_model(model_config)
        self._rng = np.random.default_rng(seed)
        (
            self._data_diversity_score,
            self._label_distribution,
            self._class_presence_vector,
        ) = self._compute_data_profile()
        self.historical_utility = float(np.clip(0.45 + 0.35 * self._data_diversity_score, 0.0, 1.0))

    def simulate_round_conditions(self):
        battery_drain = self._rng.integers(1, 5)
        latency_shift = self._rng.integers(-8, 9)

        self.battery_level = int(np.clip(self.battery_level - battery_drain, 15, 100))
        self.network_latency = int(np.clip(self.network_latency + latency_shift, 5, 200))
        self.rounds_since_last_selection += 1

    def selection_freshness_score(self):
        return float(min(self.rounds_since_last_selection / 4.0, 1.0))

    def selection_pressure(self):
        if self.selection_count == 0:
            return 0.0
        return float(self.accepted_update_count / self.selection_count)

    def system_quality_score(self):
        battery_term = self.battery_level / 100.0
        latency_term = max(0.0, 1.0 - (self.network_latency / 200.0))
        data_term = min(self.data_size / 900.0, 1.0)
        return (
            0.35 * battery_term
            + 0.20 * latency_term
            + 0.30 * data_term
            + 0.15 * self.reliability_score
        )

    def quality_score(self):
        return (
            0.30 * self.system_quality_score()
            + 0.25 * self.data_diversity_score()
            + 0.25 * self.historical_utility
            + 0.20 * self.reliability_score
        )

    def data_diversity_score(self):
        return self._data_diversity_score

    def label_distribution(self):
        return self._label_distribution

    def class_presence_vector(self):
        return self._class_presence_vector

    def _compute_data_profile(self):
        labels = self.train_y.detach().cpu().numpy()
        class_counts = np.bincount(labels, minlength=self.model_config.num_classes).astype(float)
        class_probs = class_counts / max(class_counts.sum(), 1.0)
        non_zero = class_probs[class_probs > 0]
        entropy = -np.sum(non_zero * np.log(non_zero))
        max_entropy = np.log(self.model_config.num_classes) if self.model_config.num_classes > 1 else 1.0
        entropy_score = float(entropy / max(max_entropy, 1e-8))
        feature_variance = float(torch.var(self.train_x).item())
        variance_score = float(np.tanh(feature_variance))
        diversity_score = 0.7 * entropy_score + 0.3 * variance_score
        class_presence = (class_counts > 0).astype(float)
        return diversity_score, class_probs, class_presence

    def mark_selected(self):
        self.selection_count += 1
        self.rounds_since_last_selection = 0

    def update_after_round(self, update_metrics, accepted):
        learning_signal = (
            0.45 * update_metrics["accuracy"]
            + 0.25 * float(np.exp(-update_metrics["loss"]))
            + 0.20 * self.data_diversity_score()
            + 0.10 * self.system_quality_score()
        )
        if accepted:
            self.accepted_update_count += 1
            reliability_target = 0.55 + 0.45 * update_metrics["accuracy"]
            self.historical_utility = float(
                np.clip(0.7 * self.historical_utility + 0.3 * learning_signal, 0.05, 1.0)
            )
            self.reliability_score = float(
                np.clip(0.8 * self.reliability_score + 0.2 * reliability_target, 0.2, 1.0)
            )
        else:
            self.rejected_update_count += 1
            penalized_signal = max(0.05, 0.55 * learning_signal)
            self.historical_utility = float(
                np.clip(0.82 * self.historical_utility + 0.18 * penalized_signal, 0.05, 1.0)
            )
            self.reliability_score = float(np.clip(self.reliability_score * 0.82, 0.15, 1.0))

    def selection_state_vector(self):
        return {
            "battery_level": self.battery_level,
            "network_latency": self.network_latency,
            "data_size": self.data_size,
            "freshness_score": self.selection_freshness_score(),
            "diversity_score": self.data_diversity_score(),
            "historical_utility": self.historical_utility,
            "reliability_score": self.reliability_score,
            "selection_count": self.selection_count,
            "accepted_update_count": self.accepted_update_count,
            "rejected_update_count": self.rejected_update_count,
        }

    def _apply_attack(self, delta, attack_type, attack_scale):
        if attack_type == "sign_flip":
            return [-attack_scale * layer for layer in delta]
        if attack_type == "gaussian_noise":
            attacked_delta = []
            for layer in delta:
                noise = self._rng.normal(loc=0.0, scale=attack_scale, size=layer.shape)
                attacked_delta.append(layer + noise.astype(layer.dtype))
            return attacked_delta
        return delta

    def train_local_model(
        self,
        global_weights,
        epochs=2,
        lr=0.02,
        attack_config=None,
        verbose=True,
    ):
        """Train locally and return a transmitted update."""
        set_model_weights(self.local_model, global_weights)
        base_weights = clone_weights(global_weights)

        if verbose:
            print(
                f"   [Device {self.client_id}] Training on {self.data_size} samples "
                f"(battery={self.battery_level}%, latency={self.network_latency}ms)..."
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.local_model.parameters(), lr=lr)
        batch_size = int(min(64, max(16, self.data_size // 4)))
        estimated_train_steps = int(np.ceil(self.data_size / batch_size) * epochs)

        self.local_model.train()
        for _ in range(epochs):
            permutation = torch.randperm(self.data_size)
            for batch_start in range(0, self.data_size, batch_size):
                batch_indices = permutation[batch_start : batch_start + batch_size]
                batch_x = self.train_x[batch_indices]
                batch_y = self.train_y[batch_indices]

                optimizer.zero_grad()
                outputs = self.local_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        updated_weights = get_model_weights(self.local_model)
        delta = compute_weight_delta(base_weights, updated_weights)

        is_adversarial = bool(attack_config and attack_config.get("enabled"))
        if is_adversarial:
            delta = self._apply_attack(
                delta,
                attack_type=attack_config.get("attack_type", "sign_flip"),
                attack_scale=attack_config.get("attack_scale", 5.0),
            )
            updated_weights = apply_weight_delta(base_weights, delta)

        metrics = self.evaluate(weights=updated_weights)
        learning_quality = (
            0.45 * metrics["accuracy"]
            + 0.25 * float(np.exp(-metrics["loss"]))
            + 0.20 * self.data_diversity_score()
            + 0.10 * self.historical_utility
        )
        transmitted_quality = (
            0.30 * self.system_quality_score()
            + 0.45 * learning_quality
            + 0.15 * self.data_diversity_score()
            + 0.10 * self.reliability_score
        )
        delta_norm = float(
            np.linalg.norm(np.concatenate([layer.reshape(-1) for layer in delta]))
        )
        upload_scalars = int(sum(layer.size for layer in delta))
        energy_proxy = (
            (upload_scalars / 1000.0)
            * (1.0 + self.network_latency / 100.0)
            * (1.1 - self.battery_level / 100.0)
            * (1.0 + delta_norm)
        )
        train_time_proxy_ms = float(
            estimated_train_steps
            * (0.18 * self.model_config.hidden_size + 0.04 * self.model_config.input_size)
        )

        return {
            "client_id": self.client_id,
            "weights": updated_weights,
            "delta": delta,
            "num_samples": self.data_size,
            "quality_score": float(transmitted_quality),
            "utility_score": float(learning_quality),
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "latency_ms": float(self.network_latency),
            "energy_proxy": float(energy_proxy),
            "train_time_proxy_ms": train_time_proxy_ms,
            "upload_scalars": upload_scalars,
            "battery_level": self.battery_level,
            "is_adversarial": is_adversarial,
            "reliability_score": self.reliability_score,
            "diversity_score": self.data_diversity_score(),
            "freshness_score": self.selection_freshness_score(),
        }

    def evaluate(self, weights=None, split="eval"):
        if weights is not None:
            set_model_weights(self.local_model, weights)

        if split == "train":
            data_x, data_y = self.train_x, self.train_y
        else:
            data_x, data_y = self.eval_x, self.eval_y

        self.local_model.eval()
        with torch.no_grad():
            outputs = self.local_model(data_x)
            loss = nn.CrossEntropyLoss()(outputs, data_y).item()
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == data_y).float().mean().item()

        return {"loss": loss, "accuracy": accuracy}
