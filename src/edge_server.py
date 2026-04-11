import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model import SimpleModel, get_model_weights
from smart_aggregator import SmartAggregator


class EdgeServer:
    def __init__(self, edge_id, random_state=42):
        self.edge_id = edge_id
        self.aggregator = SmartAggregator()
        self.selection_model = self._build_selection_model(random_state=random_state)

    def _build_selection_model(self, random_state):
        rng = np.random.default_rng(random_state)
        training_examples = []
        labels = []

        for _ in range(300):
            battery = rng.integers(10, 101)
            latency = rng.integers(5, 201)
            data_size = rng.integers(100, 1001)

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

    def intelligent_client_selection(self, clients, required_clients=5):
        """
        Select clients using a learned ranking model over device conditions.
        """
        print(f"\n[Edge {self.edge_id}] Running intelligent client selection...")
        scored_clients = []

        for client in clients:
            client.simulate_round_conditions()
            features = np.array(
                [[client.battery_level, client.network_latency, client.data_size]]
            )
            probability = self.selection_model.predict_proba(features)[0][1]
            scored_clients.append((probability, client))
            print(
                f" - Client {client.client_id}: score={probability:.2f}, "
                f"battery={client.battery_level}%, latency={client.network_latency}ms, "
                f"data={client.data_size}"
            )

        scored_clients.sort(key=lambda item: item[0], reverse=True)
        selected_clients = [client for _, client in scored_clients[:required_clients]]
        print(f"Selected Clients: {[client.client_id for client in selected_clients]}")
        return selected_clients

    def process_round(
        self,
        clients,
        incoming_global_weights=None,
        required_clients=3,
        round_num=None,
    ):
        """
        Run one complete edge round: selection, local training, filtering, aggregation.
        """
        if incoming_global_weights is None:
            incoming_global_weights = get_model_weights(SimpleModel())

        selected_clients = self.intelligent_client_selection(
            clients, required_clients=required_clients
        )

        client_updates = []
        for client in selected_clients:
            update = client.train_local_model(incoming_global_weights)
            client_updates.append(update)

        safe_updates = self.aggregator.detect_outliers(client_updates)
        edge_summary = self.aggregator.aggregate(safe_updates, incoming_global_weights)

        if edge_summary is None:
            return None

        print(
            f"[Edge {self.edge_id}] Round summary: "
            f"clients={edge_summary['client_ids']}, "
            f"total_samples={edge_summary['num_samples']}, "
            f"avg_accuracy={edge_summary['avg_accuracy']:.4f}, "
            f"avg_loss={edge_summary['avg_loss']:.4f}"
        )

        return {
            "edge_id": self.edge_id,
            "weights": edge_summary["weights"],
            "num_samples": edge_summary["num_samples"],
            "num_clients": len(edge_summary["client_ids"]),
            "avg_accuracy": edge_summary["avg_accuracy"],
            "avg_loss": edge_summary["avg_loss"],
        }
