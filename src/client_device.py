import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import (
    SimpleModel,
    clone_weights,
    compute_weight_delta,
    get_model_weights,
    set_model_weights,
)


class DeviceClient:
    def __init__(self, client_id, battery_level, network_latency, data_size, seed):
        self.client_id = client_id
        self.battery_level = battery_level
        self.network_latency = network_latency
        self.data_size = data_size
        self.seed = seed

        self.local_model = SimpleModel()
        self._rng = np.random.default_rng(seed)
        self.data_x, self.data_y = self._build_local_dataset()

    def _build_local_dataset(self):
        feature_count = 10
        center = self._rng.normal(0.0, 1.0, size=feature_count)
        data_x = self._rng.normal(loc=center, scale=1.0, size=(self.data_size, feature_count))

        signal = (
            0.9 * data_x[:, 0]
            - 0.6 * data_x[:, 1]
            + 0.4 * data_x[:, 2]
            + 0.3 * data_x[:, 3]
            + self._rng.normal(0.0, 0.8, size=self.data_size)
        )
        data_y = (signal > 0).astype(np.int64)

        return (
            torch.tensor(data_x, dtype=torch.float32),
            torch.tensor(data_y, dtype=torch.long),
        )

    def simulate_round_conditions(self):
        battery_drain = self._rng.integers(1, 5)
        latency_shift = self._rng.integers(-8, 9)

        self.battery_level = int(np.clip(self.battery_level - battery_drain, 15, 100))
        self.network_latency = int(np.clip(self.network_latency + latency_shift, 5, 200))

    def quality_score(self):
        battery_term = self.battery_level / 100.0
        latency_term = max(0.0, 1.0 - (self.network_latency / 200.0))
        data_term = min(self.data_size / 900.0, 1.0)
        return 0.40 * battery_term + 0.25 * latency_term + 0.35 * data_term

    def train_local_model(self, global_weights, epochs=2, lr=0.02):
        """Train on local data starting from the current global model."""
        set_model_weights(self.local_model, global_weights)
        base_weights = clone_weights(global_weights)

        print(
            f"   [Device {self.client_id}] Training locally on {self.data_size} samples "
            f"(battery={self.battery_level}%, latency={self.network_latency}ms)..."
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.local_model.parameters(), lr=lr)

        self.local_model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.local_model(self.data_x)
            loss = criterion(outputs, self.data_y)
            loss.backward()
            optimizer.step()

        updated_weights = get_model_weights(self.local_model)
        delta = compute_weight_delta(base_weights, updated_weights)
        metrics = self.evaluate(weights=updated_weights)

        return {
            "client_id": self.client_id,
            "weights": updated_weights,
            "delta": delta,
            "num_samples": self.data_size,
            "quality_score": self.quality_score(),
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
        }

    def evaluate(self, weights=None):
        if weights is not None:
            set_model_weights(self.local_model, weights)

        self.local_model.eval()
        with torch.no_grad():
            outputs = self.local_model(self.data_x)
            loss = nn.CrossEntropyLoss()(outputs, self.data_y).item()
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == self.data_y).float().mean().item()

        return {"loss": loss, "accuracy": accuracy}
