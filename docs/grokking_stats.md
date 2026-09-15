# Grokking Stats and Changepoint Detection

This document explains the statistical methods used to automatically detect and analyze the "grokking cliff" across different data-collapse severities.

## Changepoint Detection

Historically, the "grokking step" was eyeballed from accuracy curves. We replaced this manual process with a robust, automated changepoint detection algorithm:

1. **Sigmoid Curve Fitting**: We fit a logistic curve $L / (1 + \exp(-k(x - x_0))) + b$ to the test-accuracy trajectory using `scipy.optimize.curve_fit`. The parameter $x_0$ represents the transition center (the actual grokking step).
2. **Derivative Baseline**: As a fallback and comparison, we implemented a maximum 2nd derivative detector (`max_2nd_derivative_detector`) which uses central differences over a sliding window to identify the steepest part of the learning curve.
3. **Uncertainty Estimates**: The curve-fitting algorithm computes a covariance matrix for the parameters, giving us standard errors and a quantifiable "width" of the transition zone ($\approx 4/k$).

These methods are validated on synthetic step-function and noisy-cliff curves to ensure their reliability.

## Survival Analysis

Because severe data-collapse can prevent grokking entirely within the 50,000-step limit, we frame the grokking step as a "time-to-event" variable:

- **Censoring**: If a model fails to grok by the end of training (`max_steps`), its grokking time is right-censored. We track this explicitly as `event_observed=False`.
- **Kaplan-Meier Estimator**: We compute the probability of surviving (not yet grokking) over training steps for each severity condition.
- **Log-Rank / Permutation Tests**: To robustly compare conditions, we implement permutation tests for differences in grokking times and Hedge's $g$ effect size calculations (with 95% Confidence Intervals) to measure the magnitude of the delay caused by data collapse.

## Multi-seed Aggregation

The script `src/analysis/grok_detector/run_aggregator.py` groups the outputs from the multi-seed infrastructure (`results/multi_seed/<seed>/<condition>`). It pulls the final `grokking_step` and formats the results for both effect size computation and Kaplan-Meier curves.

## Code Fixes

During this update, we corrected a few legacy bugs:
- Fixed a sliding-window artifact in `compute_learning_speed` that shifted the apparent cliff relative to the true test-accuracy jump.
- Standardized the grokking accuracy threshold to 0.95 across all analysis and detection scripts.
- Resolved off-by-one errors in log-step extraction to guarantee exact alignment between the training loop and the analysis toolchain.
