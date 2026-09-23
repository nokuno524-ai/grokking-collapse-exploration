"""
Statistical tests for grokking cliff comparisons.
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy import stats

def mann_whitney_u_test(
    group1: List[float],
    group2: List[float]
) -> Tuple[float, float]:
    """
    Perform a Mann-Whitney U test between two groups.
    Returns (statistic, p_value).
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    # Filter out NaNs if any
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    if len(g1) == 0 or len(g2) == 0:
        return float('nan'), float('nan')

    stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    return float(stat), float(p_val)

def holm_correction(p_values: List[float]) -> List[float]:
    """
    Apply Holm's step-down procedure for multiple comparisons correction.
    """
    p_vals = np.array(p_values)
    n = len(p_vals)

    # Handle NaNs
    valid_mask = ~np.isnan(p_vals)
    valid_p = p_vals[valid_mask]

    if len(valid_p) == 0:
        return p_values

    # Sort p-values
    sorted_indices = np.argsort(valid_p)
    sorted_p = valid_p[sorted_indices]

    # Apply Holm correction
    corrected_p = np.minimum(1.0, sorted_p * (len(valid_p) - np.arange(len(valid_p))))

    # Enforce monotonicity
    for i in range(1, len(corrected_p)):
        corrected_p[i] = max(corrected_p[i], corrected_p[i-1])

    # Map back to original order
    # Create an array of nans with the size of original valid_p before filling
    res_valid = np.zeros(len(valid_p))
    for i, idx in enumerate(sorted_indices):
        res_valid[idx] = corrected_p[i]

    res = np.full(n, np.nan)
    res[valid_mask] = res_valid

    return res.tolist()

def permutation_test(
    group1: List[float],
    group2: List[float],
    n_permutations: int = 10000
) -> float:
    """
    Perform a permutation test for the difference in means.
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    if len(g1) == 0 or len(g2) == 0:
        return float('nan')

    observed_diff = np.abs(np.mean(g1) - np.mean(g2))

    combined = np.concatenate([g1, g2])
    n1 = len(g1)

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        diff = np.abs(np.mean(combined[:n1]) - np.mean(combined[n1:]))
        if diff >= observed_diff:
            count += 1

    return count / n_permutations

def hedges_g(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Hedge's g effect size.
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)

    if n1 < 2 or n2 < 2:
        return float('nan')

    mean1, mean2 = np.mean(g1), np.mean(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if s_pooled == 0:
        return 0.0

    # Cohen's d
    d = (mean1 - mean2) / s_pooled

    # Hedge's correction factor
    j = 1 - 3 / (4 * (n1 + n2) - 9)

    return d * j

def bootstrap_ci(
    data: List[float],
    ci: float = 0.95,
    n_resamples: int = 10000
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence intervals for the mean.
    """
    arr = np.array(data)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 2:
        return float('nan'), float('nan')

    means = []
    n = len(arr)

    for _ in range(n_resamples):
        sample = np.random.choice(arr, size=n, replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)

    return float(lower), float(upper)
