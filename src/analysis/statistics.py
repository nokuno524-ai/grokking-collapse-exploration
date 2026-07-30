import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any, Optional, Union
import pandas as pd

def bootstrap_ci(data: np.ndarray, num_bootstrap: int = 1000, ci: float = 95.0, statistic: callable = np.mean) -> Tuple[float, float, float]:
    """
    Computes the bootstrap confidence interval for a given statistic.

    Args:
        data: 1D numpy array of data points.
        num_bootstrap: Number of bootstrap samples.
        ci: Confidence interval level (e.g., 95.0).
        statistic: Function to compute the statistic (default: np.mean).

    Returns:
        Tuple of (statistic_value, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    base_stat = statistic(data)

    # Generate bootstrap samples
    bootstrap_samples = np.random.choice(data, size=(num_bootstrap, len(data)), replace=True)
    bootstrap_stats = np.array([statistic(sample) for sample in bootstrap_samples])

    # Calculate percentiles
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    return float(base_stat), float(lower_bound), float(upper_bound)

def permutation_test_grokking(
    grokking_steps_a: np.ndarray,
    grokking_steps_b: np.ndarray,
    num_permutations: int = 10000
) -> Tuple[float, float]:
    """
    Perform a permutation test to determine if the difference in mean grokking steps
    between two conditions is statistically significant.

    Args:
        grokking_steps_a: Grokking steps for condition A (e.g., pure).
        grokking_steps_b: Grokking steps for condition B (e.g., collapsed).
        num_permutations: Number of permutations.

    Returns:
        Tuple of (observed_difference, p_value)
    """
    # Remove NaNs (cases that didn't grok) or fill them with a max value before passing to this function
    a = np.asarray(grokking_steps_a)
    b = np.asarray(grokking_steps_b)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan

    obs_diff = float(np.mean(a) - np.mean(b))

    combined = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(num_permutations):
        permuted = np.random.permutation(combined)
        perm_a = permuted[:n_a]
        perm_b = permuted[n_a:]
        perm_diff = np.mean(perm_a) - np.mean(perm_b)

        # Two-tailed test
        if abs(perm_diff) >= abs(obs_diff):
            count += 1

    p_value = count / num_permutations
    return obs_diff, float(p_value)

def benjamini_hochberg(p_values: List[float], fdr_level: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Apply Benjamini-Hochberg procedure for multiple comparison correction.

    Args:
        p_values: List of p-values.
        fdr_level: False Discovery Rate level.

    Returns:
        Tuple of (significant_flags, adjusted_p_values)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    if n == 0:
        return [], []

    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    adjusted_p_values = np.zeros(n)

    # Calculate adjusted p-values
    min_adj_p = 1.0
    for i in range(n - 1, -1, -1):
        adj_p = sorted_p_values[i] * n / (i + 1)
        min_adj_p = min(min_adj_p, adj_p)
        adjusted_p_values[sorted_indices[i]] = min_adj_p

    # Cap at 1.0
    adjusted_p_values = np.minimum(adjusted_p_values, 1.0)

    significant = adjusted_p_values <= fdr_level

    return significant.tolist(), adjusted_p_values.tolist()

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Args:
        group_a: Data for group A.
        group_b: Data for group B.

    Returns:
        Cohen's d value.
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    n_a = len(a)
    n_b = len(b)

    if n_a < 2 or n_b < 2:
        return np.nan

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    # Pooled standard deviation
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_sd == 0:
        return np.nan

    d = (np.mean(a) - np.mean(b)) / pooled_sd
    return float(d)
