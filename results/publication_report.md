# Publication Benchmark Report

## Headline Findings

- Best final accuracy: `hierarchical_intelligent_no_filter` at 0.8872 +- 0.0196
- Best macro-F1: `hierarchical_intelligent_no_filter` at 0.8847
- Lowest latency proxy: `hierarchical_intelligent_fedavg` at 113.02 ms
- Best fairness: `hierarchical_intelligent_no_filter` at 0.8993
- Lowest participation inequality: `proposed` at Gini=0.1797
- Strongest worst-class accuracy: `hierarchical_intelligent_no_filter` at 0.5934
- Best attack detection: `flat_fedavg_random` at 0.0000

## Baseline Comparison

- Strongest hierarchical method vs flat FedAvg: `hierarchical_intelligent_no_filter` improves macro-F1 by 0.0186
- Cloud upload reduction vs flat FedAvg: 75.00%
- Worst-class accuracy change vs flat FedAvg: 0.0188

## Method Rankings

1. `hierarchical_intelligent_no_filter`: accuracy=0.8872, macro_f1=0.8847, worst_class=0.5934, bytes=964000, fairness=0.8993, gini=0.1802, attack_detect=0.0000
2. `hierarchical_intelligent_fedavg`: accuracy=0.8750, macro_f1=0.8720, worst_class=0.5833, bytes=964000, fairness=0.5219, gini=0.5227, attack_detect=0.0000
3. `flat_fedavg_random`: accuracy=0.8717, macro_f1=0.8661, worst_class=0.5746, bytes=771200, fairness=0.8804, gini=0.2027, attack_detect=0.0000
4. `hierarchical_intelligent_no_fairness`: accuracy=0.8733, macro_f1=0.8645, worst_class=0.4529, bytes=964000, fairness=0.5354, gini=0.5068, attack_detect=0.0000
5. `proposed`: accuracy=0.8644, macro_f1=0.8583, worst_class=0.4319, bytes=964000, fairness=0.8985, gini=0.1797, attack_detect=0.0000
6. `hierarchical_random_fedavg`: accuracy=0.8533, macro_f1=0.8381, worst_class=0.4288, bytes=964000, fairness=0.8544, gini=0.2310, attack_detect=0.0000

## Suggested Paper Tables

- Table 1: Final accuracy, macro-F1, balanced accuracy, worst-class accuracy, and bytes.
- Table 2: Fairness metrics including Jain fairness, participation entropy, coverage, and Gini.
- Table 3: Security metrics including attack detection, attack escape rate, benign retention, and filter precision.
- Figure 1: Accuracy and macro-F1 by round with seed variance bands.
- Figure 2: Communication and latency trade-offs across methods.
- Figure 3: Fairness and participation inequality comparison across methods.
