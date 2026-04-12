# Publication Benchmark Report

## Headline Findings

- Best final accuracy: `proposed` at 0.8269 +- 0.0380
- Lowest latency proxy: `hierarchical_intelligent_fedavg` at 107.25 ms
- Best fairness: `flat_fedavg_random` at 0.8574
- Best attack detection: `hierarchical_intelligent_no_fairness` at 1.0000

## Method Rankings

1. `proposed`: accuracy=0.8269, gain=0.6657, bytes=771200, fairness=0.7080, convergence_round=5.67
2. `hierarchical_intelligent_no_fairness`: accuracy=0.7741, gain=0.5787, bytes=771200, fairness=0.4637, convergence_round=5.33
3. `hierarchical_random_fedavg`: accuracy=0.1167, gain=-0.0093, bytes=771200, fairness=0.8124, convergence_round=4.33
4. `hierarchical_intelligent_no_filter`: accuracy=0.1074, gain=0.0093, bytes=771200, fairness=0.6973, convergence_round=1.33
5. `flat_fedavg_random`: accuracy=0.0972, gain=-0.0824, bytes=616960, fairness=0.8574, convergence_round=1.67
6. `hierarchical_intelligent_fedavg`: accuracy=0.0963, gain=-0.0046, bytes=771200, fairness=0.4614, convergence_round=2.00

## Ablation Guide

- `flat_fedavg_random` is the classic flat baseline.
- `hierarchical_random_fedavg` removes intelligent client selection.
- `hierarchical_intelligent_fedavg` removes quality-weighted aggregation.
- `hierarchical_intelligent_no_filter` removes anomaly filtering.
- `hierarchical_intelligent_no_fairness` removes fairness-aware participation balancing.
