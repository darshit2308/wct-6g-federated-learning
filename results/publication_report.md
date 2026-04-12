# Publication Benchmark Report

## Headline Findings

- Best final accuracy: `hierarchical_intelligent_no_fairness` at 0.8491 +- 0.0284
- Lowest latency proxy: `hierarchical_intelligent_fedavg` at 105.92 ms
- Best fairness: `flat_fedavg_random` at 0.8574
- Best attack detection: `flat_fedavg_random` at 0.0000

## Method Rankings

1. `hierarchical_intelligent_no_fairness`: accuracy=0.8491, gain=0.6222, bytes=771200, fairness=0.4604, convergence_round=6.67
2. `hierarchical_random_fedavg`: accuracy=0.8361, gain=0.4185, bytes=771200, fairness=0.8536, convergence_round=5.67
3. `hierarchical_intelligent_fedavg`: accuracy=0.8259, gain=0.5944, bytes=771200, fairness=0.4572, convergence_round=7.00
4. `hierarchical_intelligent_no_filter`: accuracy=0.8241, gain=0.6481, bytes=771200, fairness=0.6831, convergence_round=5.33
5. `flat_fedavg_random`: accuracy=0.8241, gain=0.5694, bytes=616960, fairness=0.8574, convergence_round=5.00
6. `proposed`: accuracy=0.8000, gain=0.5750, bytes=771200, fairness=0.6906, convergence_round=5.33

## Ablation Guide

- `flat_fedavg_random` is the classic flat baseline.
- `hierarchical_random_fedavg` removes intelligent client selection.
- `hierarchical_intelligent_fedavg` removes quality-weighted aggregation.
- `hierarchical_intelligent_no_filter` removes anomaly filtering.
- `hierarchical_intelligent_no_fairness` removes fairness-aware participation balancing.
