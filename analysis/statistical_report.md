# Statistical Analysis of Grokking Cliffs

This report analyzes the effect of model collapse severity on the grokking cliff.
A logistic curve $y = \text{bottom} + \frac{\text{top} - \text{bottom}}{1 + e^{-k(x - x_0)}}$ is fit to the test accuracy vs. step curve for each run.

## Cliff Statistics by Severity

| Severity | N (Valid) | Grokking Step (95% CI) | Cliff Width (95% CI) | Asymptotic Acc (95% CI) | Mean R² |
|---|---|---|---|---|---|
| pure | 5 | 968 [838, 1097] | 643 [440, 845] | 0.997 [0.995, 0.998] | 0.91 |
| low_collapse | 5 | 1424 [1326, 1522] | 1575 [1342, 1808] | 0.978 [0.972, 0.984] | 0.98 |
| medium_collapse | 5 | 2757 [1987, 3526] | 5156 [3283, 7029] | 0.834 [0.793, 0.876] | 0.89 |
| high_collapse | 5 | 6861 [3537, 10184] | 35651 [12214, 59089] | 0.289 [0.242, 0.336] | 0.81 |
| severe_collapse | 0 | - | - | - | 0.00 |

## Hypothesis Tests

### Effect of Severe Collapse on Grokking Step

- Not enough valid data points to compare Pure and Severe Collapse.

### Trend Analysis Across Severity Levels

- **Spearman trend test p-value (Grokking Step across severities):** 0.0000

## Caveats

- Runs with no grokking (flat accuracy curves) are excluded from step and width calculations.
- If N < 5, confidence intervals may be unreliable.