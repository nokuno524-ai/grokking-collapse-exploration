# Grokking Multi-Seed Statistical Report

This report presents the results of the multi-seed robustness analysis for model collapse effects on grokking.

## Summary Statistics

| Condition | Seeds | Grok Rate | Mean Grok Step | 95% Final Acc | Censored |
|-----------|-------|-----------|----------------|---------------|----------|
| pure | 1 | 0.00% | nan | 0.007 | 1 |
| low_collapse | 1 | 0.00% | nan | 0.011 | 1 |
| medium_collapse | 1 | 0.00% | nan | 0.009 | 1 |
| high_collapse | 1 | 0.00% | nan | 0.011 | 1 |
| severe_collapse | 1 | 0.00% | nan | 0.013 | 1 |

## Power Note

To distinguish adjacent conditions (e.g., pure vs low_collapse) at α=0.05 with 80% power:
- Insufficient grokking events to compute effect size between `pure` and `low_collapse`.

## Visualizations
- ![Grokking Distributions](grokking_distributions.png)
- ![Survival Curves](survival_curves.png)
- ![Final Accuracies](final_accuracies.png)
