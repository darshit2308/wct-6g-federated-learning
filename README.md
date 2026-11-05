# Adaptive Hierarchical Federated Learning for Fair and Robust 6G Edge Intelligence

Official simulation codebase and benchmark suite for IEEE Networking Letters manuscript:
**NL2026-0420**: *"Adaptive Hierarchical Federated Learning for Fair and Robust 6G Edge Intelligence"*

**Authors:**
- Jay Hirapara (ABV-IIITM Gwalior)
- Darshit Khandelwal (ABV-IIITM Gwalior)
- Mahendra K. Shukla (*Senior Member, IEEE*, ABV-IIITM Gwalior)
- Manish Pratap Singh (DYSL-CT, DRDO, Chennai)
- Om Jee Pandey (*Senior Member, IEEE*, IIT BHU Varanasi)

---

## Overview

This repository implements the 3-tier hierarchical federated learning (HFL) framework for heterogeneous 6G edge intelligence:
$$\text{End-User Devices} \longrightarrow \text{Edge Base Stations (Edge Servers)} \longrightarrow \text{Central Cloud Coordinator}$$

### Key Technical Contributions
1. **Dynamic Multi-Criterion Client Selection:** Evaluates local update quality via Cosine Directional Alignment, Channel SNR, battery energy availability, and participation recency with Lyapunov drift-plus-penalty stability.
2. **Defensive Distance-Normalized Cosine Aggregation:** Edge-level defense against adversarial manipulation, featuring parameterized threshold filtering ($\tau = 0.58$) and coordinate-wise outlier rejection.
3. **Bandwidth-Efficient Asynchronous Edge-to-Cloud Uploads:** Achieves ~75% reduction in uplink communication overhead compared to flat federated learning baselines.
4. **Fairness Preservation:** Guarantees participation equity across non-IID client distributions, preventing starvation of power-constrained edge devices (Jain's Fairness Index $> 0.88$).

---

## Implemented Methods & Baselines

All 10 methods evaluated across 5 random seeds (42, 52, 62, 72, 82) under Non-IID Dirichlet ($\alpha = 0.5$) partitioning:

| # | Method Identifier | Description / Reference |
|---|-------------------|-------------------------|
| 1 | `proposed_adaptive_hfl` | **Proposed AHFL**: Multi-criterion scoring, Lyapunov drift-plus-penalty, edge defense ($\tau=0.58$) |
| 2 | `hierarchical_direct_score` | Direct-Score baseline without Lyapunov drift-plus-penalty |
| 3 | `krum` | Multi-Krum Byzantine-robust aggregation (*Blanchard et al., NeurIPS 2017*) |
| 4 | `trimmed_mean` | Coordinate-wise Trimmed Mean aggregation (*Yin et al., ICML 2018*) |
| 5 | `m3d_hfl` | Multi-tier Multi-criteria Dynamic HFL baseline |
| 6 | `hierarchical_fl` | Standard 3-tier Hierarchical FL with local edge aggregation |
| 7 | `fedprox` | Proximal regularization FL ($\mu = 0.01$) (*Li et al., MLSys 2020*) |
| 8 | `fedavg` | Canonical Federated Averaging (*McMahan et al., AISTATS 2017*) |
| 9 | `random_selection` | Flat FL with uniform random client selection |
| 10 | `greedy_snr` | Flat FL selecting exclusively top-SNR clients (starvation baseline) |

---

## Adversarial Robustness Benchmark

Evaluated against 4 distinct threat models with 20% compromised edge devices:
1. **Adaptive Fang Attack** (*Fang et al., USENIX Security 2020*): Directed coordinate manipulation targeting local model alignment.
2. **Min-Max Attack**: Bounded optimization craft maximizing parameter perturbation within benign distance constraints.
3. **Label-Flipping Attack**: Data poisoning via systematic source-to-target permutation ($y \leftarrow 9 - y$).
4. **Sign-Flipping / Gradient Inversion Attack**: Directional inversion ($\Delta w \leftarrow -\sigma \Delta w$).

---

## Repository Structure

```text
wct-6g-federated-learning/
├── main.py                          # Single-run CLI entrypoint
├── run_master_benchmark.py          # 5-seed comprehensive benchmark runner
├── sensitivity_analysis.py          # Parameter sensitivity sweep (tau in [0.45, 0.72])
├── plot_pareto_frontier.py          # Trade-off evaluation (Accuracy vs Uplink Bytes)
├── generate_publication_figures.py  # Generates all 6 publication figures (300 DPI)
├── generate_final_latex_tables.py   # Formats results into publication LaTeX tables
├── requirements.txt                 # Python dependencies
├── src/
│   ├── client_device.py             # Client simulation, local training, attack models
│   ├── edge_server.py               # Edge clustering, client selection, defensive aggregation
│   ├── cloud_server.py              # Global server coordinator
│   ├── smart_aggregator.py          # Defensive cosine, Krum, Trimmed Mean algorithms
│   ├── data_utils.py                # Dirichlet Non-IID data partitioner
│   ├── experiment_runner.py         # Multi-method benchmark harness with Mean +/- Std
│   └── model.py                     # Neural network architectures (CNN, MLP)
├── results/                         # Clean benchmark CSVs and summary reports
├── results_attack/                  # Adversarial benchmark outputs
├── results_sensitivity/             # Threshold sensitivity outputs
├── results_pareto/                  # Pareto frontier outputs
└── docs/                            # Research paper, figures, rebuttal dossier
    ├── research_paper.tex           # IEEEtran LaTeX source
    ├── IEEE_Networking_Letters_Revision_Dossier.md
    └── *.png                        # All publication figures
```

---

## Quickstart & Reproduction Guide

### 1. Installation
```bash
git clone https://github.com/darshit2308/wct-6g-federated-learning.git
cd wct-6g-federated-learning
pip install -r requirements.txt
```

### 2. Run Comprehensive 5-Seed Benchmark
```bash
python run_master_benchmark.py
```
This executes all 10 algorithms across 5 seeds and saves aggregated outputs to `results/summary.csv` and `results_attack/combined_attack_summary.csv`.

### 3. Run Parameter Sensitivity & Pareto Frontier
```bash
# Threshold tau sweep:
python sensitivity_analysis.py

# Pareto frontier trade-off:
python plot_pareto_frontier.py
```

### 4. Regenerate All Publication Figures & LaTeX Tables
```bash
python generate_publication_figures.py
python generate_final_latex_tables.py
```

---

## Citation & Contact
For inquiries regarding this codebase or the manuscript revisions, please contact Jay Hirapara (`jjayhirapara2506@gmail.com`) or Dr. Mahendra K. Shukla (`mahendra@iiitm.ac.in`).
