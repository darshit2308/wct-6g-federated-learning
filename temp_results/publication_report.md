# Publication Benchmark Report

## Headline Findings

- Best final accuracy: `hierarchical_intelligent_no_fairness` at 0.3556 +- 0.0000
- Best macro-F1: `hierarchical_intelligent_no_fairness` at 0.2939
- Lowest latency proxy: `hierarchical_random_fedavg` at 93.00 ms
- Best fairness: `flat_fedavg_random` at 0.8000
- Lowest participation inequality: `flat_fedavg_random` at Gini=0.2000
- Strongest worst-class accuracy: `flat_fedavg_random` at 0.0000
- Best attack detection: `flat_fedavg_random` at 0.0000

## Baseline Comparison

- Strongest hierarchical method vs flat FedAvg: `hierarchical_intelligent_no_fairness` improves macro-F1 by 0.0324
- Cloud upload reduction vs flat FedAvg: 50.00%
- Worst-class accuracy change vs flat FedAvg: 0.0000

## Method Rankings

1. `hierarchical_intelligent_no_fairness`: accuracy=0.3556, macro_f1=0.2939, worst_class=0.0000, bytes=115680, fairness=0.6400, gini=0.3750, attack_detect=0.0000
2. `hierarchical_intelligent_fedavg`: accuracy=0.3500, macro_f1=0.2900, worst_class=0.0000, bytes=115680, fairness=0.6400, gini=0.3750, attack_detect=0.0000
3. `flat_fedavg_random`: accuracy=0.3833, macro_f1=0.2615, worst_class=0.0000, bytes=77120, fairness=0.8000, gini=0.2000, attack_detect=0.0000
4. `hierarchical_random_fedavg`: accuracy=0.4028, macro_f1=0.2568, worst_class=0.0000, bytes=115680, fairness=0.6400, gini=0.3750, attack_detect=0.0000
5. `hierarchical_intelligent_no_filter`: accuracy=0.2167, macro_f1=0.1447, worst_class=0.0000, bytes=115680, fairness=0.6400, gini=0.3750, attack_detect=0.0000
6. `proposed`: accuracy=0.2167, macro_f1=0.1447, worst_class=0.0000, bytes=115680, fairness=0.6400, gini=0.3750, attack_detect=0.0000

## Suggested Paper Tables

- Table 1: Final accuracy, macro-F1, balanced accuracy, worst-class accuracy, and bytes.
- Table 2: Fairness metrics including Jain fairness, participation entropy, coverage, and Gini.
- Table 3: Security metrics including attack detection, attack escape rate, benign retention, and filter precision.
- Figure 1: Accuracy and macro-F1 by round with seed variance bands.
- Figure 2: Communication and latency trade-offs across methods.
- Figure 3: Fairness and participation inequality comparison across methods.
