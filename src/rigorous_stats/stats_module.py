"""
Statistical routines for grokking and interpretability analysis.
Includes bootstrap confidence intervals, permutation tests, and power analysis.
"""

from typing import Tuple, List, Union
import numpy as np

def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    statistic=np.mean,
    n_resamples: int = 9999,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute a bootstrap confidence interval for a given statistic.

    Args:
        data: The sample data (1D array).
        statistic: The statistic to compute (e.g., np.mean, np.median).
        n_resamples: Number of bootstrap samples.
        confidence_level: The confidence level (default: 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        A tuple (lower_bound, upper_bound).
    """
    data = np.asarray(data)
    if len(data) == 0:
        return (np.nan, np.nan)

    rng = np.random.RandomState(random_state)
    # Generate random indices
    indices = rng.randint(0, len(data), size=(n_resamples, len(data)))
    samples = data[indices]

    # Compute statistic for each sample
    # Using apply_along_axis is safe for general functions, though slower than vectorized
    if statistic == np.mean:
        stat_dist = np.mean(samples, axis=1)
    elif statistic == np.median:
        stat_dist = np.median(samples, axis=1)
    else:
        stat_dist = np.apply_along_axis(statistic, 1, samples)

    # Calculate percentiles
    alpha = 1.0 - confidence_level
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    return np.percentile(stat_dist, [lower_pct, upper_pct])

def permutation_test(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    n_resamples: int = 9999,
    random_state: int = 42
) -> float:
    """
    Perform a two-sided permutation test to check if the means of x and y differ significantly.

    Args:
        x: First sample.
        y: Second sample.
        n_resamples: Number of permutations.
        random_state: Random seed.

    Returns:
        The empirical p-value.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.RandomState(random_state)

    observed_diff = np.abs(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)

    count_extreme = 0
    for _ in range(n_resamples):
        shuffled = rng.permutation(pooled)
        x_fake = shuffled[:n_x]
        y_fake = shuffled[n_x:]
        fake_diff = np.abs(np.mean(x_fake) - np.mean(y_fake))
        if fake_diff >= observed_diff:
            count_extreme += 1

    # Add 1 to numerator and denominator to avoid p=0 (standard practice for permutation tests)
    return (count_extreme + 1) / (n_resamples + 1)

def format_stat_result(data: Union[List[float], np.ndarray], name: str = "Metric") -> str:
    """
    Format a standardized summary string containing n, mean, and 95% CI.
    """
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return f"{name}: n=0"

    mean_val = np.mean(data)
    ci_lower, ci_upper = bootstrap_ci(data)

    return f"{name}: n={n}, mean={mean_val:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
