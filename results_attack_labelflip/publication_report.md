# Publication Benchmark Report

## Headline Findings

- Best final accuracy: `proposed` at 0.8861 +- 0.0191
- Best macro-F1: `proposed` at 0.8830
- Lowest latency proxy: `hierarchical_krum` at 119.42 ms
- Best fairness: `proposed` at 0.9071
- Lowest participation inequality: `proposed` at Gini=0.1737
- Strongest worst-class accuracy: `flat_fedavg_random` at 0.4917
- Best attack detection: `proposed` at 0.7400

## Baseline Comparison

- Strongest hierarchical method vs flat FedAvg: `proposed` improves macro-F1 by 0.0420
- Cloud upload reduction vs flat FedAvg: 75.00%
- Worst-class accuracy change vs flat FedAvg: -0.0113

## Method Rankings

1. `proposed`: accuracy=0.8861, macro_f1=0.8830, worst_class=0.4803, bytes=1924000, fairness=0.9071, gini=0.1737, attack_detect=0.7400
2. `hierarchical_intelligent_no_filter`: accuracy=0.8722, macro_f1=0.8556, worst_class=0.4526, bytes=1924000, fairness=0.8700, gini=0.2050, attack_detect=0.0000
3. `flat_fedavg_random`: accuracy=0.8467, macro_f1=0.8410, worst_class=0.4917, bytes=1539200, fairness=0.8804, gini=0.2027, attack_detect=0.0000
4. `hierarchical_trimmed_mean`: accuracy=0.8444, macro_f1=0.8346, worst_class=0.4654, bytes=1924000, fairness=0.8485, gini=0.2352, attack_detect=0.0000
5. `hierarchical_krum`: accuracy=0.8061, macro_f1=0.7907, worst_class=0.2690, bytes=1924000, fairness=0.8485, gini=0.2352, attack_detect=0.0000

## Suggested Paper Tables

- Table 1: Final accuracy, macro-F1, balanced accuracy, worst-class accuracy, and bytes.
- Table 2: Fairness metrics including Jain fairness, participation entropy, coverage, and Gini.
- Table 3: Security metrics including attack detection, attack escape rate, benign retention, and filter precision.
- Figure 1: Accuracy and macro-F1 by round with seed variance bands.
- Figure 2: Communication and latency trade-offs across methods.
- Figure 3: Fairness and participation inequality comparison across methods.
