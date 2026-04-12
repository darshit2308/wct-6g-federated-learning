import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

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

        predictions_np = predictions.detach().cpu().numpy()
        evaluation_y_np = evaluation_y.detach().cpu().numpy()
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            evaluation_y_np,
            predictions_np,
            average="macro",
            zero_division=0,
        )
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            evaluation_y_np,
            predictions_np,
            average="weighted",
            zero_division=0,
        )
        balanced_accuracy = balanced_accuracy_score(evaluation_y_np, predictions_np)

        class_accuracies = []
        for class_index in range(self.model_config.num_classes):
            class_mask = evaluation_y_np == class_index
            if not np.any(class_mask):
                class_accuracies.append(0.0)
                continue
            class_accuracies.append(
                float(np.mean(predictions_np[class_mask] == evaluation_y_np[class_mask]))
            )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "balanced_accuracy": float(balanced_accuracy),
            "worst_class_accuracy": float(min(class_accuracies)),
            "class_accuracy_std": float(np.std(class_accuracies)),
            "num_samples": total_samples,
        }
