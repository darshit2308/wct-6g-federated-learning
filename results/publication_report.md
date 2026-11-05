# Publication Benchmark Report

## Headline Findings

- Best final accuracy: `hierarchical_intelligent_no_filter` at 0.9189 +- 0.0098
- Best macro-F1: `hierarchical_intelligent_no_filter` at 0.9185
- Lowest latency proxy: `hierarchical_intelligent_fedavg` at 113.92 ms
- Best fairness: `hierarchical_fairness_direct_score` at 0.8978
- Lowest participation inequality: `hierarchical_fairness_direct_score` at Gini=0.1820
- Strongest worst-class accuracy: `hierarchical_intelligent_no_filter` at 0.7718
- Attack benchmark: not active in this run

## Baseline Comparison

- Strongest hierarchical method vs flat FedAvg: `hierarchical_intelligent_no_filter` improves macro-F1 by 0.0141
- Cloud upload reduction vs flat FedAvg: 75.00%
- Worst-class accuracy change vs flat FedAvg: 0.1006

## Method Rankings

1. `hierarchical_intelligent_no_filter`: accuracy=0.9189, macro_f1=0.9185, worst_class=0.7718, bytes=1924000, fairness=0.8955, gini=0.1830
2. `hierarchical_fairness_direct_score`: accuracy=0.9111, macro_f1=0.9099, worst_class=0.6871, bytes=1924000, fairness=0.8978, gini=0.1820
3. `hierarchical_intelligent_no_fairness`: accuracy=0.9078, macro_f1=0.9055, worst_class=0.6857, bytes=1924000, fairness=0.5429, gini=0.5032
4. `flat_fedavg_random`: accuracy=0.9061, macro_f1=0.9044, worst_class=0.6712, bytes=1539200, fairness=0.8804, gini=0.2027
5. `hierarchical_intelligent_fedavg`: accuracy=0.9039, macro_f1=0.9027, worst_class=0.7032, bytes=1924000, fairness=0.5151, gini=0.5287
6. `hierarchical_direct_score`: accuracy=0.9050, macro_f1=0.9014, worst_class=0.6032, bytes=1924000, fairness=0.5015, gini=0.5377
7. `proposed`: accuracy=0.8983, macro_f1=0.8965, worst_class=0.6496, bytes=1924000, fairness=0.8806, gini=0.2047
8. `hierarchical_trimmed_mean`: accuracy=0.8872, macro_f1=0.8839, worst_class=0.5882, bytes=1924000, fairness=0.8544, gini=0.2310
9. `hierarchical_random_fedavg`: accuracy=0.8839, macro_f1=0.8771, worst_class=0.5395, bytes=1924000, fairness=0.8544, gini=0.2310
10. `hierarchical_krum`: accuracy=0.8200, macro_f1=0.8093, worst_class=0.3439, bytes=1924000, fairness=0.8544, gini=0.2310

## Suggested Paper Tables

- Table 1: Final accuracy, macro-F1, balanced accuracy, worst-class accuracy, and bytes.
- Table 2: Fairness metrics including Jain fairness, participation entropy, coverage, and Gini.
- Table 3: Fairness or communication ablations, since no attack benchmark was active in this run.
- Figure 1: Accuracy and macro-F1 by round with seed variance bands.
- Figure 2: Communication and latency trade-offs across methods.
- Figure 3: Fairness and participation inequality comparison across methods.
