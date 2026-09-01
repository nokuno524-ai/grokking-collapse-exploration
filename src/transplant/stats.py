"""
Statistical utilities for circuit transplant experiments and replication analysis.

Includes bootstrap confidence intervals, effect size calculation (Cohen's d),
and sign consistency checks for cross-seed replications.
"""
from typing import Tuple, List, Optional
import numpy as np

def bootstrap_ci(
    data: np.ndarray | List[float],
    num_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Computes the mean and bootstrap confidence interval for a 1D array of data.

    Args:
        data: 1D array or list of numerical values.
        num_bootstraps: Number of bootstrap resamples.
        alpha: Significance level (e.g., 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        A tuple (mean, lower_bound, upper_bound).
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    data_arr = np.array(data, dtype=float)
    n = len(data_arr)

    if n == 0:
        return np.nan, np.nan, np.nan
    elif n == 1:
        val = float(data_arr[0])
        return val, val, val

    # Generate bootstrap samples
    # boot_samples shape: (num_bootstraps, n)
    boot_samples = rng.choice(data_arr, size=(num_bootstraps, n), replace=True)
    boot_means = np.mean(boot_samples, axis=1)

    # Calculate confidence interval
    lower_bound = float(np.percentile(boot_means, (alpha / 2) * 100))
    upper_bound = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    return float(np.mean(data_arr)), lower_bound, upper_bound

def cohens_d(
    group1: np.ndarray | List[float],
    group2: np.ndarray | List[float]
) -> float:
    """
    Computes Cohen's d effect size between two groups.

    Args:
        group1: First group of numerical values.
        group2: Second group of numerical values.

    Returns:
        Cohen's d value. Returns NaN if standard deviation is zero or either group is empty.
    """
    arr1 = np.array(group1, dtype=float)
    arr2 = np.array(group2, dtype=float)

    if len(arr1) == 0 or len(arr2) == 0:
        return np.nan

    n1, n2 = len(arr1), len(arr2)
    var1, var2 = np.var(arr1, ddof=1) if n1 > 1 else 0.0, np.var(arr2, ddof=1) if n2 > 1 else 0.0

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if n1 + n2 > 2 else 0.0

    if pooled_std == 0:
        return np.nan

    return float((np.mean(arr1) - np.mean(arr2)) / pooled_std)

def check_sign_consistency(
    deltas: np.ndarray | List[float]
) -> Tuple[bool, float]:
    """
    Checks if all non-zero deltas have the same sign and returns consistency ratio.

    Args:
        deltas: 1D array or list of numerical differences.

    Returns:
        A tuple (is_consistent, consistency_ratio).
        consistency_ratio is the proportion of elements that share the majority sign
        (ignoring exact zeros).
    """
    arr = np.array(deltas, dtype=float)
    non_zeros = arr[arr != 0]

    if len(non_zeros) == 0:
        return True, 1.0

    positive_count = np.sum(non_zeros > 0)
    negative_count = np.sum(non_zeros < 0)

    majority_count = max(positive_count, negative_count)
    total_non_zeros = len(non_zeros)

    consistency_ratio = float(majority_count / total_non_zeros)
    is_consistent = (consistency_ratio == 1.0)

    return is_consistent, consistency_ratio
