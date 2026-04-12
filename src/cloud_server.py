import numpy as np
import torch
import torch.nn as nn

from model import build_model, get_model_weights, set_model_weights, weighted_average


class CloudServer:
    def __init__(self, model_config):
        self.model_config = model_config
        self.global_model = build_model(model_config)
        self.global_weights = get_model_weights(self.global_model)

    def hierarchical_aggregation(self, edge_updates, verbose=True):
        """Combine edge models into the final global model using sample weighting."""
        if not edge_updates:
            return self.global_weights

        valid_edge_updates = [update for update in edge_updates if update["num_samples"] > 0]
        if not valid_edge_updates:
            return self.global_weights

        if verbose:
            print("\n[CLOUD LAYER] Performing final global aggregation across all edge servers...")

        total_samples = sum(update["num_samples"] for update in valid_edge_updates)
        coefficients = [update["num_samples"] / total_samples for update in valid_edge_updates]
        edge_weight_sets = [update["weights"] for update in valid_edge_updates]

        self.global_weights = weighted_average(edge_weight_sets, coefficients)
        set_model_weights(self.global_model, self.global_weights)

        if verbose:
            avg_edge_accuracy = np.mean([update["avg_accuracy"] for update in valid_edge_updates])
            avg_edge_loss = np.mean([update["avg_loss"] for update in valid_edge_updates])
            print(
                f"[CLOUD LAYER] Global model updated successfully. "
                f"Edge Avg Accuracy={avg_edge_accuracy:.4f}, Edge Avg Loss={avg_edge_loss:.4f}"
            )
        return self.global_weights

    def flat_fedavg_aggregation(self, client_updates, verbose=True):
        """Traditional FedAvg over selected client models."""
        if not client_updates:
            return self.global_weights

        if verbose:
            print("\n[CLOUD LAYER] Performing flat FedAvg aggregation across selected clients...")

        total_samples = sum(update["num_samples"] for update in client_updates)
        coefficients = [update["num_samples"] / total_samples for update in client_updates]
        weight_sets = [update["weights"] for update in client_updates]

        self.global_weights = weighted_average(weight_sets, coefficients)
        set_model_weights(self.global_model, self.global_weights)
        return self.global_weights

    def evaluate_global_model(self, evaluation_x, evaluation_y):
        total_samples = len(evaluation_y)

        set_model_weights(self.global_model, self.global_weights)
        self.global_model.eval()

        with torch.no_grad():
            outputs = self.global_model(evaluation_x)
            loss = nn.CrossEntropyLoss()(outputs, evaluation_y).item()
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == evaluation_y).float().mean().item()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "num_samples": total_samples,
        }
