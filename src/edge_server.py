import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model import build_model, get_model_weights
from smart_aggregator import SmartAggregator


class EdgeServer:
    def __init__(
        self,
        edge_id,
        model_config,
        random_state=42,
        selection_strategy="intelligent",
        aggregation_strategy="quality_weighted",
        use_outlier_filter=True,
        local_epochs=2,
        learning_rate=0.02,
    ):
        self.edge_id = edge_id
        self.model_config = model_config
        self.selection_strategy = selection_strategy
        self.aggregation_strategy = aggregation_strategy
        self.use_outlier_filter = use_outlier_filter
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.aggregator = SmartAggregator()
        self.selection_model = self._build_selection_model(random_state=random_state)
        self._rng = np.random.default_rng(random_state)

    def _adjusted_selection_score(self, client, probability):
        fairness_bonus = 1.0 / (1.0 + client.selection_count)
        diversity_bonus = client.data_diversity_score()
        return (
            0.55 * probability
            + 0.20 * diversity_bonus
            + 0.15 * fairness_bonus
            + 0.10 * client.reliability_score
        )

    def _build_selection_model(self, random_state):
        rng = np.random.default_rng(random_state)
        training_examples = []
        labels = []

        for _ in range(500):
            battery = rng.integers(10, 101)
            latency = rng.integers(5, 201)
            data_size = rng.integers(100, 5001)

            rule_score = (
                0.45 * (battery / 100.0)
                + 0.35 * (1.0 - latency / 200.0)
                + 0.20 * (data_size / 1000.0)
            )

            training_examples.append([battery, latency, data_size])
            labels.append(1 if rule_score >= 0.58 else 0)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=random_state)),
            ]
        )
        model.fit(np.array(training_examples), np.array(labels))
        return model

    def advance_round_state(self, clients):
        for client in clients:
            client.simulate_round_conditions()

    def select_clients(self, clients, required_clients=5, verbose=True):
        if self.selection_strategy == "random":
            selected_clients = list(
                self._rng.choice(clients, size=required_clients, replace=False)
            )
            if verbose:
                print(f"\n[Edge {self.edge_id}] Running random client selection...")
                print(f"Selected Clients: {[client.client_id for client in selected_clients]}")
            return selected_clients

        if verbose:
            print(f"\n[Edge {self.edge_id}] Running intelligent client selection...")

        scored_clients = []
        for client in clients:
            features = np.array(
                [[client.battery_level, client.network_latency, client.data_size]]
            )
            probability = self.selection_model.predict_proba(features)[0][1]
            adjusted_score = self._adjusted_selection_score(client, probability)
            scored_clients.append((adjusted_score, client, probability))
            if verbose:
                print(
                    f" - Client {client.client_id}: score={adjusted_score:.2f}, "
                    f"base_prob={probability:.2f}, "
                    f"battery={client.battery_level}%, latency={client.network_latency}ms, "
                    f"data={client.data_size}, diversity={client.data_diversity_score():.2f}, "
                    f"selected_before={client.selection_count}"
                )

        scored_clients.sort(key=lambda item: item[0], reverse=True)
        selected_clients = [client for _, client, _ in scored_clients[:required_clients]]
        if verbose:
            print(f"Selected Clients: {[client.client_id for client in selected_clients]}")
        return selected_clients

    def process_round(
        self,
        clients,
        incoming_global_weights=None,
        required_clients=3,
        round_num=None,
        attack_fraction=0.0,
        attack_type="sign_flip",
        attack_scale=5.0,
        verbose=True,
    ):
        """
        Run one complete edge round: selection, local training, filtering, aggregation.
        """
        if incoming_global_weights is None:
            incoming_global_weights = get_model_weights(build_model(self.model_config))

        self.advance_round_state(clients)
        selected_clients = self.select_clients(
            clients, required_clients=required_clients, verbose=verbose
        )

        if not selected_clients:
            return None

        num_adversarial = int(round(len(selected_clients) * attack_fraction))
        adversarial_ids = set()
        if num_adversarial > 0:
            adversarial_ids = set(
                client.client_id
                for client in self._rng.choice(
                    selected_clients, size=num_adversarial, replace=False
                )
            )

        client_updates = []
        for client in selected_clients:
            client.selection_count += 1
            update = client.train_local_model(
                incoming_global_weights,
                epochs=self.local_epochs,
                lr=self.learning_rate,
                attack_config={
                    "enabled": client.client_id in adversarial_ids,
                    "attack_type": attack_type,
                    "attack_scale": attack_scale,
                },
                verbose=verbose,
            )
            client_updates.append(update)

        edge_summary = self.aggregator.aggregate(
            client_updates,
            incoming_global_weights,
            aggregation_strategy=self.aggregation_strategy,
            use_outlier_filter=self.use_outlier_filter,
            verbose=verbose,
        )

        if edge_summary is None:
            return None

        if verbose:
            print(
                f"[Edge {self.edge_id}] Round summary: "
                f"clients={edge_summary['client_ids']}, "
                f"total_samples={edge_summary['num_samples']}, "
                f"avg_accuracy={edge_summary['avg_accuracy']:.4f}, "
                f"avg_loss={edge_summary['avg_loss']:.4f}"
            )

        edge_summary.update(
            {
                "edge_id": self.edge_id,
                "client_uploads": len(selected_clients),
                "cloud_uploads": 1,
                "payload_to_edge_scalars": edge_summary["payload_scalars"],
                "payload_to_cloud_scalars": int(
                    sum(layer.size for layer in edge_summary["weights"])
                ),
            }
        )
        return edge_summary
