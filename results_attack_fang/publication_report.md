# Publication Benchmark Report

## Headline Findings

- Best final accuracy: `proposed` at 0.8933 +- 0.0314
- Best macro-F1: `proposed` at 0.8915
- Lowest latency proxy: `proposed` at 119.30 ms
- Best fairness: `proposed` at 0.8994
- Lowest participation inequality: `proposed` at Gini=0.1827
- Strongest worst-class accuracy: `proposed` at 0.6538
- Best attack detection: `proposed` at 0.7800

## Baseline Comparison

- Strongest hierarchical method vs flat FedAvg: `proposed` improves macro-F1 by 0.0193
- Cloud upload reduction vs flat FedAvg: 75.00%
- Worst-class accuracy change vs flat FedAvg: 0.1259

## Method Rankings

1. `proposed`: accuracy=0.8933, macro_f1=0.8915, worst_class=0.6538, bytes=1924000, fairness=0.8994, gini=0.1827, attack_detect=0.7800
2. `flat_fedavg_random`: accuracy=0.8772, macro_f1=0.8722, worst_class=0.5278, bytes=1539200, fairness=0.8804, gini=0.2027, attack_detect=0.0000
3. `hierarchical_trimmed_mean`: accuracy=0.8706, macro_f1=0.8617, worst_class=0.4556, bytes=1924000, fairness=0.8485, gini=0.2352, attack_detect=0.0000
4. `hierarchical_intelligent_no_filter`: accuracy=0.8622, macro_f1=0.8510, worst_class=0.4451, bytes=1924000, fairness=0.8862, gini=0.1922, attack_detect=0.0000
5. `hierarchical_krum`: accuracy=0.8061, macro_f1=0.7901, worst_class=0.2619, bytes=1924000, fairness=0.8485, gini=0.2352, attack_detect=0.0000

## Suggested Paper Tables

- Table 1: Final accuracy, macro-F1, balanced accuracy, worst-class accuracy, and bytes.
- Table 2: Fairness metrics including Jain fairness, participation entropy, coverage, and Gini.
- Table 3: Security metrics including attack detection, attack escape rate, benign retention, and filter precision.
- Figure 1: Accuracy and macro-F1 by round with seed variance bands.
- Figure 2: Communication and latency trade-offs across methods.
- Figure 3: Fairness and participation inequality comparison across methods.
