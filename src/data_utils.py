from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits

from client_device import DeviceClient
from model import ModelConfig


@dataclass
class DatasetBundle:
    clients: list
    evaluation_x: torch.Tensor
    evaluation_y: torch.Tensor
    model_config: ModelConfig
    dataset_name: str


def _random_device_state(rng):
    battery_level = int(rng.integers(45, 101))
    network_latency = int(rng.integers(10, 121))
    reliability_score = float(rng.uniform(0.8, 1.0))
    return battery_level, network_latency, reliability_score


def _make_synthetic_client_tensors(train_size, eval_size, input_size, seed):
    rng = np.random.default_rng(seed)
    center = rng.normal(0.0, 1.0, size=input_size)
    coefficients = np.zeros(input_size)
    base_coefficients = np.array([0.9, -0.6, 0.4, 0.3])
    coefficients[: min(len(base_coefficients), input_size)] = base_coefficients[
        : min(len(base_coefficients), input_size)
    ]
    if input_size > len(base_coefficients):
        coefficients[len(base_coefficients) :] = rng.normal(
            0.0, 0.2, size=input_size - len(base_coefficients)
        )

    def build_split(size):
        data_x = rng.normal(loc=center, scale=1.0, size=(size, input_size))
        signal = data_x @ coefficients + rng.normal(0.0, 0.8, size=size)
        data_y = (signal > 0).astype(np.int64)
        return (
            torch.tensor(data_x, dtype=torch.float32),
            torch.tensor(data_y, dtype=torch.long),
        )

    return build_split(train_size), build_split(eval_size)


def _dirichlet_partition(labels, num_clients, alpha, rng, min_size):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    for _ in range(50):
        client_indices = [[] for _ in range(num_clients)]
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            rng.shuffle(label_indices)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            split_points = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
            for client_index, chunk in enumerate(np.split(label_indices, split_points)):
                client_indices[client_index].extend(chunk.tolist())

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= min_size:
            for indices in client_indices:
                rng.shuffle(indices)
            return client_indices
    raise RuntimeError("Could not build a stable Dirichlet partition for the dataset.")


def build_dataset_bundle(
    dataset_name,
    num_clients,
    seed,
    hidden_size,
    synthetic_feature_count=10,
    synthetic_min_samples=240,
    synthetic_max_samples=900,
    eval_ratio=0.2,
    dataset_root="data",
    download_dataset=False,
    dirichlet_alpha=0.5,
):
    rng = np.random.default_rng(seed)

    if dataset_name == "synthetic":
        clients = []
        eval_x_parts = []
        eval_y_parts = []
        model_config = ModelConfig(
            input_size=synthetic_feature_count,
            hidden_size=hidden_size,
            num_classes=2,
        )

        for client_index in range(num_clients):
            train_size = int(rng.integers(synthetic_min_samples, synthetic_max_samples + 1))
            eval_size = max(40, int(train_size * eval_ratio))
            (train_x, train_y), (eval_x, eval_y) = _make_synthetic_client_tensors(
                train_size=train_size,
                eval_size=eval_size,
                input_size=synthetic_feature_count,
                seed=seed + client_index,
            )
            battery_level, network_latency, reliability_score = _random_device_state(rng)
            clients.append(
                DeviceClient(
                    client_id=f"C{client_index + 1}",
                    battery_level=battery_level,
                    network_latency=network_latency,
                    train_x=train_x,
                    train_y=train_y,
                    eval_x=eval_x,
                    eval_y=eval_y,
                    seed=seed + client_index,
                    model_config=model_config,
                    reliability_score=reliability_score,
                )
            )
            eval_x_parts.append(eval_x)
            eval_y_parts.append(eval_y)

        return DatasetBundle(
            clients=clients,
            evaluation_x=torch.cat(eval_x_parts, dim=0),
            evaluation_y=torch.cat(eval_y_parts, dim=0),
            model_config=model_config,
            dataset_name=dataset_name,
        )

    if dataset_name == "digits":
        digits = load_digits()
        data_x = torch.tensor(digits.data / 16.0, dtype=torch.float32)
        data_y = torch.tensor(digits.target, dtype=torch.long)

        all_indices = np.arange(len(data_y))
        rng.shuffle(all_indices)
        split_at = int(0.8 * len(all_indices))
        train_indices = all_indices[:split_at]
        eval_indices = all_indices[split_at:]

        train_x = data_x[train_indices]
        train_y = data_y[train_indices]
        eval_x = data_x[eval_indices]
        eval_y = data_y[eval_indices]

        train_partitions = _dirichlet_partition(
            train_y.numpy(),
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            rng=rng,
            min_size=30,
        )
        eval_partitions = _dirichlet_partition(
            eval_y.numpy(),
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            rng=rng,
            min_size=8,
        )

        clients = []
        model_config = ModelConfig(input_size=64, hidden_size=hidden_size, num_classes=10)
        global_eval_x_parts = []
        global_eval_y_parts = []

        for client_index in range(num_clients):
            client_train_indices = train_partitions[client_index]
            client_eval_indices = eval_partitions[client_index]
            battery_level, network_latency, reliability_score = _random_device_state(rng)
            client = DeviceClient(
                client_id=f"C{client_index + 1}",
                battery_level=battery_level,
                network_latency=network_latency,
                train_x=train_x[client_train_indices],
                train_y=train_y[client_train_indices],
                eval_x=eval_x[client_eval_indices],
                eval_y=eval_y[client_eval_indices],
                seed=seed + client_index,
                model_config=model_config,
                reliability_score=reliability_score,
            )
            clients.append(client)
            global_eval_x_parts.append(client.eval_x)
            global_eval_y_parts.append(client.eval_y)

        return DatasetBundle(
            clients=clients,
            evaluation_x=torch.cat(global_eval_x_parts, dim=0),
            evaluation_y=torch.cat(global_eval_y_parts, dim=0),
            model_config=model_config,
            dataset_name=dataset_name,
        )

    if dataset_name == "mnist":
        try:
            from torchvision import datasets, transforms
        except ImportError as exc:
            raise RuntimeError(
                "torchvision is required for the MNIST benchmark but is not installed."
            ) from exc

        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(
            root=dataset_root,
            train=True,
            download=download_dataset,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            root=dataset_root,
            train=False,
            download=download_dataset,
            transform=transform,
        )

        train_x = train_dataset.data.float().view(-1, 28 * 28) / 255.0
        train_y = train_dataset.targets.long()
        eval_x = test_dataset.data.float().view(-1, 28 * 28) / 255.0
        eval_y = test_dataset.targets.long()

        train_partitions = _dirichlet_partition(
            train_y.numpy(),
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            rng=rng,
            min_size=200,
        )
        eval_partitions = _dirichlet_partition(
            eval_y.numpy(),
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            rng=rng,
            min_size=40,
        )

        clients = []
        model_config = ModelConfig(input_size=784, hidden_size=hidden_size, num_classes=10)
        global_eval_x_parts = []
        global_eval_y_parts = []

        for client_index in range(num_clients):
            client_train_indices = train_partitions[client_index]
            client_eval_indices = eval_partitions[client_index]
            battery_level, network_latency, reliability_score = _random_device_state(rng)
            client = DeviceClient(
                client_id=f"C{client_index + 1}",
                battery_level=battery_level,
                network_latency=network_latency,
                train_x=train_x[client_train_indices],
                train_y=train_y[client_train_indices],
                eval_x=eval_x[client_eval_indices],
                eval_y=eval_y[client_eval_indices],
                seed=seed + client_index,
                model_config=model_config,
                reliability_score=reliability_score,
            )
            clients.append(client)
            global_eval_x_parts.append(client.eval_x)
            global_eval_y_parts.append(client.eval_y)

        return DatasetBundle(
            clients=clients,
            evaluation_x=torch.cat(global_eval_x_parts, dim=0),
            evaluation_y=torch.cat(global_eval_y_parts, dim=0),
            model_config=model_config,
            dataset_name=dataset_name,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")
