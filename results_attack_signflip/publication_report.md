# Publication Benchmark Report

## Headline Findings

- Best final accuracy: `proposed` at 0.8950 +- 0.0211
- Best macro-F1: `proposed` at 0.8923
- Lowest latency proxy: `proposed` at 119.12 ms
- Best fairness: `hierarchical_intelligent_no_filter` at 0.8886
- Lowest participation inequality: `hierarchical_intelligent_no_filter` at Gini=0.1887
- Strongest worst-class accuracy: `proposed` at 0.6090
- Best attack detection: `proposed` at 1.0000

## Baseline Comparison

- Strongest hierarchical method vs flat FedAvg: `proposed` improves macro-F1 by 0.8735
- Cloud upload reduction vs flat FedAvg: 75.00%
- Worst-class accuracy change vs flat FedAvg: 0.6090

## Method Rankings

1. `proposed`: accuracy=0.8950, macro_f1=0.8923, worst_class=0.6090, bytes=1924000, fairness=0.8882, gini=0.1910, attack_detect=1.0000
2. `hierarchical_krum`: accuracy=0.7889, macro_f1=0.7658, worst_class=0.2333, bytes=1924000, fairness=0.8485, gini=0.2352, attack_detect=0.0000
3. `hierarchical_intelligent_no_filter`: accuracy=0.1411, macro_f1=0.0417, worst_class=0.0000, bytes=1924000, fairness=0.8886, gini=0.1887, attack_detect=0.0000
4. `hierarchical_trimmed_mean`: accuracy=0.1094, macro_f1=0.0290, worst_class=0.0000, bytes=1924000, fairness=0.8485, gini=0.2352, attack_detect=0.0000
5. `flat_fedavg_random`: accuracy=0.0978, macro_f1=0.0188, worst_class=0.0000, bytes=1539200, fairness=0.8804, gini=0.2027, attack_detect=0.0000

## Suggested Paper Tables

- Table 1: Final accuracy, macro-F1, balanced accuracy, worst-class accuracy, and bytes.
- Table 2: Fairness metrics including Jain fairness, participation entropy, coverage, and Gini.
- Table 3: Security metrics including attack detection, attack escape rate, benign retention, and filter precision.
- Figure 1: Accuracy and macro-F1 by round with seed variance bands.
- Figure 2: Communication and latency trade-offs across methods.
- Figure 3: Fairness and participation inequality comparison across methods.
