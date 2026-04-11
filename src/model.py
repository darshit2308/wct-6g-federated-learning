import copy

import numpy as np
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """A lightweight neural network used in the FL simulation."""

    def __init__(self, input_size=10, hidden_size=32, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_model_weights(model):
    """Extract weights from a PyTorch model as detached NumPy arrays."""
    return [tensor.detach().cpu().numpy().copy() for tensor in model.state_dict().values()]


def set_model_weights(model, weights):
    """Load a NumPy weight list into a PyTorch model."""
    state_dict = copy.deepcopy(model.state_dict())
    for key, weight in zip(state_dict.keys(), weights):
        state_dict[key] = torch.tensor(weight, dtype=state_dict[key].dtype)
    model.load_state_dict(state_dict)


def clone_weights(weights):
    return [layer.copy() for layer in weights]


def compute_weight_delta(base_weights, updated_weights):
    return [updated - base for base, updated in zip(base_weights, updated_weights)]


def apply_weight_delta(base_weights, delta):
    return [base + layer_delta for base, layer_delta in zip(base_weights, delta)]


def weighted_average(weight_sets, coefficients):
    if not weight_sets:
        return None

    averaged = [np.zeros_like(layer) for layer in weight_sets[0]]
    for coefficient, weight_set in zip(coefficients, weight_sets):
        for layer_index, layer in enumerate(weight_set):
            averaged[layer_index] += layer * coefficient
    return averaged
