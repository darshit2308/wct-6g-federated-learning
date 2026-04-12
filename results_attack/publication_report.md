# Publication Benchmark Report

## Headline Findings

- Best final accuracy: `proposed` at 0.8756 +- 0.0222
- Best macro-F1: `proposed` at 0.8697
- Lowest latency proxy: `hierarchical_intelligent_fedavg` at 112.84 ms
- Best fairness: `proposed` at 0.9047
- Lowest participation inequality: `proposed` at Gini=0.1770
- Strongest worst-class accuracy: `proposed` at 0.4962
- Best attack detection: `hierarchical_intelligent_no_fairness` at 1.0000

## Baseline Comparison

- Strongest hierarchical method vs flat FedAvg: `proposed` improves macro-F1 by 0.8513
- Cloud upload reduction vs flat FedAvg: 75.00%
- Worst-class accuracy change vs flat FedAvg: 0.4962

## Method Rankings

1. `proposed`: accuracy=0.8756, macro_f1=0.8697, worst_class=0.4962, bytes=964000, fairness=0.9047, gini=0.1770, attack_detect=1.0000
2. `hierarchical_intelligent_no_fairness`: accuracy=0.8422, macro_f1=0.8295, worst_class=0.3868, bytes=964000, fairness=0.5980, gini=0.4522, attack_detect=1.0000
3. `hierarchical_intelligent_no_filter`: accuracy=0.1150, macro_f1=0.0279, worst_class=0.0000, bytes=964000, fairness=0.8957, gini=0.1832, attack_detect=0.0000
4. `hierarchical_intelligent_fedavg`: accuracy=0.1017, macro_f1=0.0186, worst_class=0.0000, bytes=964000, fairness=0.5763, gini=0.4732, attack_detect=0.0000
5. `flat_fedavg_random`: accuracy=0.1011, macro_f1=0.0183, worst_class=0.0000, bytes=771200, fairness=0.8804, gini=0.2027, attack_detect=0.0000
6. `hierarchical_random_fedavg`: accuracy=0.0917, macro_f1=0.0168, worst_class=0.0000, bytes=964000, fairness=0.8485, gini=0.2352, attack_detect=0.0000

## Suggested Paper Tables

- Table 1: Final accuracy, macro-F1, balanced accuracy, worst-class accuracy, and bytes.
- Table 2: Fairness metrics including Jain fairness, participation entropy, coverage, and Gini.
- Table 3: Security metrics including attack detection, attack escape rate, benign retention, and filter precision.
- Figure 1: Accuracy and macro-F1 by round with seed variance bands.
- Figure 2: Communication and latency trade-offs across methods.
- Figure 3: Fairness and participation inequality comparison across methods.
