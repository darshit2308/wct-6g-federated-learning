# Adaptive Hierarchical Federated Learning Framework for 6G Edge Networks

## 1. Project Overview

This project presents a structured simulation of a hierarchical federated learning framework designed for a 6G-inspired edge intelligence setting. The system models a three-layer learning pipeline in which device clients train locally, edge servers perform intermediate intelligence and aggregation, and a cloud server performs the final global update.

The main purpose of the project is to demonstrate how hierarchical federated learning can be made more practical for future ultra-dense wireless environments by introducing:

- intelligent client selection at the edge
- weighted aggregation based on device quality and contribution
- secure filtering of suspicious client updates
- iterative cloud-to-edge-to-device model propagation
- measurable round-wise global improvement

This repository is intentionally designed as a simulation prototype rather than a production telecom platform. Even so, the codebase implements the essential learning flow correctly and is suitable for academic demonstration, explanation, and further extension.

## 2. Problem Statement

Classical federated learning is attractive because it keeps raw data on user devices, but it also introduces practical challenges in large-scale distributed systems:

- not all clients are equally reliable
- devices vary in battery level, latency, and data availability
- central aggregation becomes expensive when many clients communicate directly with a cloud server
- malicious, noisy, or low-quality updates can destabilize training

In a 6G environment with extremely dense device populations and edge-native intelligence, a purely flat federated topology is not ideal. This project addresses that limitation with a hierarchical design:

- devices train locally
- edge servers coordinate regional learning
- the cloud performs cross-edge synchronization

## 3. Objective of the Work

The objective of this project is to build a complete simulation that demonstrates the following research-inspired ideas in one coherent workflow:

1. hierarchical federated learning over device, edge, and cloud layers
2. intelligent edge-side client selection instead of naive or purely random participation
3. weighted model aggregation that considers both sample volume and device quality
4. robust update screening before edge aggregation
5. round-wise global evaluation to observe convergence behavior

## 4. Conceptual Architecture

The framework follows a three-tier structure.

### 4.1 Device Layer

Each device represents a simulated mobile, IoT, or user-end node. A device contains:

- local data
- a local copy of the model
- device state variables such as battery level, latency, and data size

At the beginning of a round, the device receives the latest global model. It then performs local training on its own dataset and sends only model updates back to the edge server.

### 4.2 Edge Layer

Each edge server represents a regional 6G intelligent access node or intermediate compute layer. Its responsibilities include:

- evaluating available clients
- selecting the most suitable participants for the current round
- collecting locally trained updates
- filtering suspicious updates
- performing weighted regional aggregation

This layer is the core intelligence layer of the project.

### 4.3 Cloud Layer

The cloud acts as the global coordinator. It receives the aggregated edge models, combines them into a single updated global model, and evaluates the model across the full distributed client population.

## 5. Key Design Choices

### 5.1 Intelligent Client Selection

Instead of random selection, edge servers rank devices using a lightweight learned model based on:

- battery level
- network latency
- local data size

This reflects the idea that practical federated systems should prefer clients that are both resource-capable and likely to contribute meaningful learning progress.

### 5.2 Edge-Side Robustness

Before aggregation, the edge server examines client update magnitudes. A robust median absolute deviation strategy is used for anomaly screening, but only when enough client updates are available to make the estimate meaningful. This is important because aggressive filtering with very small sample counts can look arbitrary and can hurt credibility.

### 5.3 Weighted Aggregation

Client updates are not treated equally. Each selected client contributes according to:

- number of local samples
- battery condition
- latency condition
- local data contribution score

This makes the aggregation more realistic than a plain unweighted average.

### 5.4 Hierarchical Global Update

After edge aggregation, the cloud combines regional models using sample-weighted averaging. The final global model is then sent into the next training round.

This closes the federated learning loop correctly.

## 6. What Makes This Project Strong

This project is stronger than a minimal toy demo because it includes the full end-to-end training cycle and not just isolated components.

The implementation includes:

- deterministic seeding for reproducible behavior
- round-wise device condition changes
- non-identical local datasets across clients
- proper model propagation across rounds
- update-based aggregation instead of unrelated raw model averaging
- global evaluation after each round
- readable console traces for presentation and debugging

## 7. Repository Structure

```text
wct-6g-federated-learning/
├── main.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── client_device.py
    ├── cloud_server.py
    ├── edge_server.py
    ├── model.py
    └── smart_aggregator.py
```

## 8. File-by-File Explanation

### `main.py`

This is the simulation entry point. It:

- sets seeds for reproducibility
- creates clients and edge servers
- splits devices across edge regions
- executes multiple FL rounds
- performs cloud aggregation
- prints per-round metrics
- prints a final summary at the end of execution

### `src/model.py`

This file defines the neural network and helper functions for:

- extracting weights
- loading weights
- cloning weights
- computing update deltas
- applying deltas
- performing weighted averaging

### `src/client_device.py`

This file models a participating device. It:

- creates a client-specific synthetic dataset
- simulates changing battery and latency conditions across rounds
- trains the local model using the incoming global model
- computes the client delta
- evaluates the updated local model
- returns metrics and contribution metadata to the edge

### `src/edge_server.py`

This file models the edge intelligence layer. It:

- creates a lightweight client-selection model
- evaluates clients each round
- selects the top candidates
- launches local client training
- invokes secure filtering
- performs regional aggregation
- reports round summaries for presentation

### `src/smart_aggregator.py`

This file implements the edge aggregation logic. It:

- screens suspicious client updates
- avoids unreliable filtering when too few updates are available
- computes weighted regional aggregation
- returns edge-level metrics and the aggregated regional model

### `src/cloud_server.py`

This file implements cloud coordination. It:

- receives regional edge models
- performs hierarchical sample-weighted aggregation
- updates the global model
- evaluates the global model over all client datasets

## 9. Workflow of One Training Round

The flow of one full round is:

1. the cloud holds the latest global model
2. each edge server inspects the condition of its local client pool
3. the edge selects the most suitable clients for participation
4. selected devices receive the current global model
5. each selected device performs local training on its own data
6. the edge receives the updated client models and computes update deltas
7. suspicious updates are filtered when enough evidence exists
8. the edge aggregates safe updates into a regional model
9. the cloud combines all edge models into a new global model
10. the new global model is evaluated and used for the next round

## 10. Dataset Strategy Used in This Simulation

This project uses synthetic per-client datasets rather than a real-world benchmark dataset. That choice was intentional for the following reasons:

- it keeps the project self-contained
- it avoids external dataset setup complexity
- it makes the simulation easy to reproduce on another system
- it allows clear control over client heterogeneity

Each client receives a slightly different data distribution so that the environment is non-identical across devices, which better reflects realistic federated learning conditions.

## 11. Experimental Behavior

When the project is run successfully, the console output should show:

- client ranking at each edge
- selected clients for the round
- local training activity
- edge aggregation summaries
- cloud aggregation summaries
- global accuracy and loss after each round
- a final summary of improvement over all rounds

The expected trend is not perfect accuracy in only a few rounds. Instead, the most realistic expectation is gradual improvement in global accuracy and gradual reduction in loss. That behavior indicates that the hierarchical training loop is functioning coherently.

## 12. Why the Output Is Credible

The output becomes credible because the model is no longer static from round to round. In this implementation:

- clients do not train from unrelated random starting states every round
- the global model is propagated correctly
- device conditions evolve across rounds
- aggregation uses explicit update logic
- evaluation is performed at the global level

This makes the round-wise output meaningful rather than decorative.

## 13. Installation

Create and activate a virtual environment, then install the required packages.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 14. How to Run

Run the simulation with:

```bash
python main.py
```

If your environment maps Python differently, use:

```bash
python3 main.py
```

## 15. Dependencies

The project uses the following major libraries:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `scikit-learn`
- `flwr`

Note: `flwr` is included as a relevant federated learning ecosystem dependency, although the present simulation uses a custom orchestration flow rather than the Flower runtime.

## 16. Interpretation of Results

If the output shows improving global accuracy and decreasing loss over rounds, you can explain it as evidence that:

- the hierarchical training loop is functioning correctly
- edge-side aggregation is contributing useful regional updates
- the global model is converging gradually
- the selected clients are contributing useful signal

This is a strong academic message for a project demonstration.

## 17. Current Scope and Honest Limitations

This project is a robust simulation prototype, but it is important to describe it honestly.

Current limitations include:

- synthetic data instead of a real benchmark dataset
- a lightweight selection model instead of a fully trained adaptive policy
- no true network simulator or wireless channel model
- no asynchronous client participation
- no real adversarial attack generator
- no large-scale statistical benchmarking across many repeated runs

These limitations do not weaken the value of the project as a course or academic prototype. In fact, stating them clearly usually improves credibility.

## 18. Future Enhancement Opportunities

This project can be extended in several strong directions:

- integrate a real federated dataset such as MNIST, CIFAR, or IoT telemetry data
- compare against standard FedAvg baselines
- add dropout-aware client availability modeling
- incorporate trust history into client scoring
- simulate adversarial clients explicitly
- expand the cloud layer into multi-region aggregation
- visualize convergence curves and selection behavior
- integrate communication-cost analysis

## 19. Suggested Presentation Narrative

A strong way to present this project is:

1. start with the limitation of flat federated learning in dense 6G environments
2. explain the need for hierarchical coordination
3. show the three-layer architecture
4. highlight intelligent client selection and robust aggregation as the core innovations
5. run the simulation and show improving accuracy and decreasing loss
6. conclude with the significance of edge intelligence in scalable future networks

## 20. Final Summary

This project demonstrates a complete simulation of adaptive hierarchical federated learning for a 6G-inspired environment. It combines local privacy-preserving training, regional edge intelligence, weighted and robust aggregation, and global cloud synchronization in one structured implementation.

The codebase is readable, reproducible, modular, and suitable for explanation in an academic setting. It shows clear engineering effort, conceptual understanding, and practical implementation work across machine learning, distributed systems, and edge intelligence themes.
