import numpy as np
import torch
import torch.nn as nn

from model import SimpleModel, get_model_weights, set_model_weights, weighted_average


class CloudServer:
    def __init__(self):
        self.global_model = SimpleModel()
        self.global_weights = get_model_weights(self.global_model)

    def hierarchical_aggregation(self, edge_updates):
        """Combine edge models into the final global model using sample weighting."""
        if not edge_updates:
            return self.global_weights

        print("\n[CLOUD LAYER] Performing final global aggregation across all edge servers...")

        total_samples = sum(update["num_samples"] for update in edge_updates)
        coefficients = [update["num_samples"] / total_samples for update in edge_updates]
        edge_weight_sets = [update["weights"] for update in edge_updates]

        self.global_weights = weighted_average(edge_weight_sets, coefficients)
        set_model_weights(self.global_model, self.global_weights)

        avg_edge_accuracy = np.mean([update["avg_accuracy"] for update in edge_updates])
        avg_edge_loss = np.mean([update["avg_loss"] for update in edge_updates])
        print(
            f"[CLOUD LAYER] Global model updated successfully. "
            f"Edge Avg Accuracy={avg_edge_accuracy:.4f}, Edge Avg Loss={avg_edge_loss:.4f}"
        )
        return self.global_weights

    def evaluate_global_model(self, client_groups):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        set_model_weights(self.global_model, self.global_weights)
        self.global_model.eval()

        with torch.no_grad():
            for group in client_groups:
                for client in group:
                    outputs = self.global_model(client.data_x)
                    loss = nn.CrossEntropyLoss()(outputs, client.data_y)
                    predictions = outputs.argmax(dim=1)

                    sample_count = len(client.data_y)
                    total_loss += loss.item() * sample_count
                    total_correct += int((predictions == client.data_y).sum().item())
                    total_samples += sample_count

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
