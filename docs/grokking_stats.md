# Grokking Changepoint Detection and Statistics

This directory documents the methodology for detecting grokking cliffs and statistically comparing their timing across different training regimes. The implementation relies on validation accuracy trajectories to locate generalization transitions robustly.

## Changepoint Detection (`src/analysis/grok_detector/detectors.py`)

We employ three distinct methods to locate grokking cliffs, each serving as a check on the others:

1. **Plateau Detector (`plateau_detector`)**:
   Finds the first step where the (smoothed) accuracy curve exceeds a specified threshold (e.g., 0.95) and never falls back below it. This is the simplest and most robust definition of a grokking point.

2. **Maximum 2nd Derivative / Maximum Slope Detector (`max_2nd_derivative_detector`)**:
   Locates the inflection point of the accuracy curve. For steep grokking cliffs, this identifies the point of maximum acceleration (start of the cliff) or maximum velocity (middle of the cliff). We utilize a moving average filter to mitigate local noise before numerical differentiation.

3. **Sigmoid Fitting (`fit_sigmoid_transition`)**:
   Models the transition by fitting a generalized logistic function (sigmoid) using non-linear least squares (`scipy.optimize.curve_fit`).
   Form: `L / (1 + exp(-k * (x - x0))) + b`
   The midpoint parameter `x0` serves as the robust estimate of the cliff timing. This method is resilient to high-frequency noise but requires a well-formed transition to converge.

## Statistical Comparisons (`src/analysis/grok_detector/stats.py`)

To evaluate whether factors like dataset size, weight decay, or noise contamination significantly shift the grokking cliff, we use non-parametric statistics, owing to the often non-normal distribution of grokking steps.

- **Mann-Whitney U Test (`mann_whitney_u_test`)**: Compares the distribution of grokking steps between two conditions to determine if one consistently groks earlier or later.
- **Permutation Test (`permutation_test`)**: Calculates the exact probability of observing a difference in mean grokking steps at least as extreme as the empirical difference, assuming no underlying effect.
- **Holm-Bonferroni Correction (`holm_correction`)**: When comparing multiple conditions (e.g., a sweep over noise rates), we apply Holm's step-down procedure to control the family-wise error rate without being excessively conservative.
- **Hedge's g Effect Size (`hedges_g`)**: Quantifies the magnitude of the cliff shift between two conditions, standardized by the pooled variance, and corrected for small sample sizes.
- **Bootstrap Confidence Intervals (`bootstrap_ci`)**: Provides robust uncertainty bounds for mean grokking steps, constructed via sampling with replacement.

## Best Practices

- Always visualize the fitted sigmoids or detected plateaus on top of the raw accuracy curves to verify the detection logic.
- Ensure the evaluation cadence (e.g., `eval_every`) is fine-grained enough (e.g., 50 steps) to accurately localize the cliff without artificial quantization delays.
- Filter out `NaN` or `None` values (runs that failed to grok entirely) before conducting location-shift statistics, or encode them as right-censored data in survival analysis frameworks.
