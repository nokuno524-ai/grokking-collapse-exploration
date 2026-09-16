"""
Statistical tests for grokking.
Includes bootstrap CIs, permutation tests for collapse, and Holm-corrected Mann-Whitney U.
"""

import numpy as np
import scipy.stats as stats
from typing import Tuple, List

def bootstrap_ci(data: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Computes bootstrap confidence intervals.
    Returns (mean, lower_bound, upper_bound).
    Uses resampling with jitter to handle duplicate steps for curve fitting trajectories.
    """
    if len(data) == 0:
        return float('nan'), float('nan'), float('nan')
    means = []
    for _ in range(n_resamples):
        # Adding slight jitter to avoid identical points in curve fitting if needed
        resample = np.random.choice(data, size=len(data), replace=True)
        jitter = np.random.normal(0, 1e-6, size=len(resample))
        resampled_data = resample + jitter
        means.append(np.mean(resampled_data))

    means = np.array(means)
    return float(np.mean(data)), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))

def permutation_test_collapse(pure_metric: np.ndarray, collapse_metric: np.ndarray, n_permutations: int = 1000) -> float:
    """
    Permutation test to assess if 'collapse prevented grokking'.
    Uses a label-shuffle baseline.
    Returns p-value.
    """
    combined = np.concatenate([pure_metric, collapse_metric])
    n_pure = len(pure_metric)
    observed_diff = np.mean(pure_metric) - np.mean(collapse_metric)

    count_extreme = 0
    for _ in range(n_permutations):
        shuffled = np.random.permutation(combined)
        simulated_pure = shuffled[:n_pure]
        simulated_collapse = shuffled[n_pure:]
        sim_diff = np.mean(simulated_pure) - np.mean(simulated_collapse)
        if abs(sim_diff) >= abs(observed_diff):
            count_extreme += 1

    return count_extreme / n_permutations

def holm_mann_whitney(group1: List[float], group2: List[float]) -> float:
    """Holm-corrected Mann-Whitney U test placeholder."""
    # This is a simplified version, as true Holm correction requires multiple comparisons
    # This just returns the Mann-Whitney U p-value.
    if len(group1) == 0 or len(group2) == 0:
        return float('nan')
    _, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return p_val
