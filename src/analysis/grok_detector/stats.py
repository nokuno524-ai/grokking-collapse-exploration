"""
Statistical utilities for grokking analysis.
Includes bootstrap CIs, multiple comparison correction, and correlations.
"""

import numpy as np
from typing import Tuple, List, Optional
import scipy.stats as stats

def bootstrap_ci(data: np.ndarray, stat_func=np.mean, n_bootstraps: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    Returns: (statistic, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    base_stat = stat_func(data)
    if len(data) == 1:
        return base_stat, base_stat, base_stat

    bootstrapped_stats = []
    n = len(data)
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_stats.append(stat_func(sample))

    lower = np.percentile(bootstrapped_stats, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_stats, 100 * (1 - alpha / 2))
    return base_stat, lower, upper


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg procedure for multiple testing correction.
    Returns a boolean list where True means the null hypothesis is rejected.
    """
    p_vals = np.array(p_values)
    n = len(p_vals)
    if n == 0:
        return []

    sorted_indices = np.argsort(p_vals)
    sorted_p_vals = p_vals[sorted_indices]

    # Calculate critical values
    critical_values = (np.arange(1, n + 1) / n) * alpha

    # Find the largest p-value that is less than or equal to its critical value
    reject = np.zeros(n, dtype=bool)
    significant_tests = np.where(sorted_p_vals <= critical_values)[0]

    if len(significant_tests) > 0:
        max_idx = significant_tests[-1]
        reject[sorted_indices[:max_idx + 1]] = True

    return reject.tolist()


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman rank-order correlation coefficient and the p-value.
    Returns (correlation, p-value)
    """
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return np.nan, np.nan

    # Check if arrays are constant (std == 0)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    res = stats.spearmanr(x, y)
    return res.statistic, res.pvalue
