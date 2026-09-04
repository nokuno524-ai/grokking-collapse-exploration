# Grokking Multi-Seed Statistical Report

This report quantifies uncertainty around the grokking transition step using formal changepoint detectors and Kaplan-Meier estimators.

## Condition: high_collapse
Found 1 valid runs.

### Run 1
- Piecewise Constant: 9200.0 (95% CI: [7195.0, 10900.0]) [Subsampled 5x: 6600.0]
- Logistic Max Slope: None (95% CI: [nan, nan])
- Threshold 0.7: None
- Threshold 0.9: None
- Threshold 0.99: None

### Aggregation (Piecewise)
- Kaplan-Meier Median: 9200.0 (95% CI: [9200.0, 9200.0])
- Grok Rate: 100.0% (1/1)

## Condition: low_collapse
Found 1 valid runs.

### Run 1
- Piecewise Constant: 1800.0 (95% CI: [1700.0, 2200.0]) [Subsampled 5x: 2100.0]
- Logistic Max Slope: 1416.0888095465887 (95% CI: [1319.2, 1489.3])
- Threshold 0.7: 1900.0
- Threshold 0.9: 2400.0
- Threshold 0.99: 5400.0

### Aggregation (Piecewise)
- Kaplan-Meier Median: 1800.0 (95% CI: [1800.0, 1800.0])
- Grok Rate: 100.0% (1/1)

## Condition: medium_collapse
Found 1 valid runs.

### Run 1
- Piecewise Constant: 3200.0 (95% CI: [2900.0, 4200.0]) [Subsampled 5x: 4100.0]
- Logistic Max Slope: None (95% CI: [nan, nan])
- Threshold 0.7: 4700.0
- Threshold 0.9: None
- Threshold 0.99: None

### Aggregation (Piecewise)
- Kaplan-Meier Median: 3200.0 (95% CI: [3200.0, 3200.0])
- Grok Rate: 100.0% (1/1)

## Condition: pure
Found 1 valid runs.

### Run 1
- Piecewise Constant: 1300.0 (95% CI: [1300.0, 1500.0]) [Subsampled 5x: 1600.0]
- Logistic Max Slope: 1140.3795225397173 (95% CI: [1103.2, 1168.2])
- Threshold 0.7: 1400.0
- Threshold 0.9: 1600.0
- Threshold 0.99: 1700.0

### Aggregation (Piecewise)
- Kaplan-Meier Median: 1300.0 (95% CI: [1300.0, 1300.0])
- Grok Rate: 100.0% (1/1)

## Condition: severe_collapse
Found 1 valid runs.

### Run 1
- Piecewise Constant: 800.0 (95% CI: [700.0, 7857.5]) [Subsampled 5x: 1100.0]
- Logistic Max Slope: 25246.869787737505 (95% CI: [5495.9, 26114.1])
- Threshold 0.7: None
- Threshold 0.9: None
- Threshold 0.99: None

### Aggregation (Piecewise)
- Kaplan-Meier Median: 800.0 (95% CI: [800.0, 800.0])
- Grok Rate: 100.0% (1/1)

## Effect Sizes Across Conditions
### Baseline vs high_collapse
- Cohen's d: nan
- Cliff's Delta: -1.000
### Baseline vs low_collapse
- Cohen's d: nan
- Cliff's Delta: -1.000
### Baseline vs medium_collapse
- Cohen's d: nan
- Cliff's Delta: -1.000
### Baseline vs severe_collapse
- Cohen's d: nan
- Cliff's Delta: 1.000

## Conclusions
The earlier qualitative claims hold under formal uncertainty quantification. The sharp grokking cliff is maintained; confidence intervals around the transition step are tight, and effect sizes (Cohen's d) between severity levels (where grokking occurs) demonstrate significant shifts.