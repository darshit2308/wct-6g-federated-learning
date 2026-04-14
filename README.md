#  Adaptive Hierarchical Federated Learning (6G Edge)

## 📌 Overview
This project simulates a **hierarchical federated learning system** designed for future 6G networks.

Instead of sending all data to a central server:
- Devices train locally
- Edge servers coordinate nearby devices
- Cloud combines updates into a global model

The goal is to make federated learning more **scalable, efficient, and secure**.

---

## 🚨 Problem
Traditional federated learning has issues:
- Devices are unreliable (battery, latency, data)
- Too many clients overload the cloud
- Malicious or poor updates can harm training

---

## 💡 Solution
This project introduces a **3-layer system**:

**Device → Edge → Cloud**

With improvements:
- Smart client selection (based on device quality)
- Weighted aggregation (better devices contribute more)
- Secure filtering (removes bad updates)
- Gradual global model improvement

---

## 🧠 How It Works

Each training round:
1. Cloud sends global model
2. Edge selects best devices
3. Devices train locally
4. Devices send updates
5. Edge filters bad updates
6. Edge aggregates updates
7. Cloud updates global model

Repeat → model improves over time

---

## ⚙️ Features
- Intelligent client selection
- Weighted aggregation
- Secure update filtering
- Non-IID data simulation
- Round-wise evaluation
- Metrics tracking:
  - Accuracy
  - Macro-F1
  - Fairness
  - Security metrics

---

## 📂 Project Structure
```text
wct-6g-federated-learning/
├── main.py
├── requirements.txt
├── visualize_results.py
├── docs/
│   └── research_paper.tex
└── src/
    ├── client_device.py
    ├── edge_server.py
    ├── cloud_server.py
    ├── model.py
    ├── data_utils.py
    ├── smart_aggregator.py
    └── experiment_runner.py
