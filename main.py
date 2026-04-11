import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from client_device import DeviceClient
from cloud_server import CloudServer
from edge_server import EdgeServer


SEED = 42
NUM_CLIENTS = 20
NUM_EDGES = 2
CLIENTS_PER_EDGE = NUM_CLIENTS // NUM_EDGES
CLIENTS_PER_ROUND = 4
NUM_ROUNDS = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_clients():
    clients = []
    for client_index in range(NUM_CLIENTS):
        battery_level = random.randint(45, 100)
        network_latency = random.randint(10, 120)
        data_size = random.randint(240, 900)

        clients.append(
            DeviceClient(
                client_id=f"C{client_index + 1}",
                battery_level=battery_level,
                network_latency=network_latency,
                data_size=data_size,
                seed=SEED + client_index,
            )
        )
    return clients


def split_clients_by_edge(clients):
    return [
        clients[index : index + CLIENTS_PER_EDGE]
        for index in range(0, len(clients), CLIENTS_PER_EDGE)
    ]


def run_simulation():
    set_seed(SEED)
    print("=== Starting 6G Hierarchical FL Simulation ===")
    print(f"Seed: {SEED}")

    cloud = CloudServer()
    edges = [
        EdgeServer(edge_id="E1_North", random_state=SEED + 101),
        EdgeServer(edge_id="E2_South", random_state=SEED + 202),
    ]

    client_groups = split_clients_by_edge(create_clients())

    global_weights = cloud.global_weights
    round_history = []
    for round_num in range(1, NUM_ROUNDS + 1):
        print("\n==========================================")
        print(f"          STARTING ROUND {round_num}")
        print("==========================================")

        edge_updates = []
        for edge, edge_clients in zip(edges, client_groups):
            edge_summary = edge.process_round(
                clients=edge_clients,
                incoming_global_weights=global_weights,
                required_clients=CLIENTS_PER_ROUND,
                round_num=round_num,
            )
            if edge_summary is not None:
                edge_updates.append(edge_summary)

        global_weights = cloud.hierarchical_aggregation(edge_updates)
        global_metrics = cloud.evaluate_global_model(client_groups)

        print(
            f"[ROUND {round_num}] Global Accuracy = {global_metrics['accuracy']:.4f}, "
            f"Global Loss = {global_metrics['loss']:.4f}"
        )
        round_history.append(
            {
                "round": round_num,
                "accuracy": global_metrics["accuracy"],
                "loss": global_metrics["loss"],
            }
        )
        print(f"Round {round_num} Complete. Global model ready for next iteration.")

    print("\n==========================================")
    print("            FINAL SUMMARY")
    print("==========================================")
    for metrics in round_history:
        print(
            f"Round {metrics['round']}: "
            f"accuracy={metrics['accuracy']:.4f}, loss={metrics['loss']:.4f}"
        )

    accuracy_gain = round_history[-1]["accuracy"] - round_history[0]["accuracy"]
    loss_drop = round_history[0]["loss"] - round_history[-1]["loss"]
    print(
        f"\nOverall improvement: accuracy +{accuracy_gain:.4f}, "
        f"loss -{loss_drop:.4f}"
    )


if __name__ == "__main__":
    run_simulation()
