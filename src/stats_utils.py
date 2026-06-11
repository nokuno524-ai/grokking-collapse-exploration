import numpy as np
from typing import Tuple, List, Callable, Union

def bootstrap_ci(
    data: Union[np.ndarray, List[float]],
    statistic: Callable = np.mean,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a given statistic.

    Args:
        data: Array of data points
        statistic: Function to compute the statistic (e.g., np.mean)
        n_resamples: Number of bootstrap resamples
        ci: Confidence interval level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (statistic_value, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    n = len(data)

    # Compute the statistic on the original data
    stat_val = statistic(data)

    # Generate bootstrap samples
    indices = rng.randint(0, n, size=(n_resamples, n))
    samples = data[indices]

    # Compute statistic for each sample
    # If statistic is np.mean or similar that supports axis
    try:
        boot_stats = statistic(samples, axis=1)
    except TypeError:
        boot_stats = np.array([statistic(sample) for sample in samples])

    # Calculate percentiles
    alpha = 1.0 - ci
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    lower_bound = np.percentile(boot_stats, lower_pct)
    upper_bound = np.percentile(boot_stats, upper_pct)

    return float(stat_val), float(lower_bound), float(upper_bound)


def cohens_d(group1: Union[np.ndarray, List[float]], group2: Union[np.ndarray, List[float]]) -> float:
    """
    Compute Cohen's d for effect size between two groups.

    Args:
        group1: Data for group 1
        group2: Data for group 2

    Returns:
        Cohen's d effect size
    """
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)

    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return 0.0

    return float((np.mean(g1) - np.mean(g2)) / pooled_sd)


def bonferroni_correction(p_values: Union[np.ndarray, List[float]], alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: Array of p-values from multiple tests
        alpha: Overall significance level desired

    Returns:
        Tuple of (adjusted_p_values, corrected_alpha)
    """
    p_vals = np.asarray(p_values)
    n_tests = len(p_vals)

    corrected_alpha = alpha / n_tests
    adjusted_p_values = np.minimum(p_vals * n_tests, 1.0)

    return adjusted_p_values, corrected_alpha


def detect_phase_transition(
    series: Union[np.ndarray, List[float]],
    min_size: int = 5,
    threshold: float = 2.0
) -> int:
    """
    Detect a phase transition (change point) in a time series using a simple
    sliding window approach comparing means before and after a split point.

    Args:
        series: 1D array of values over time
        min_size: Minimum size for a segment before/after the split
        threshold: Minimum z-score difference to be considered a transition

    Returns:
        Index of the detected change point, or -1 if none found
    """
    series = np.asarray(series)
    n = len(series)

    if n < 2 * min_size:
        return -1

    max_diff = 0
    best_split = -1

    for i in range(min_size, n - min_size):
        before = series[:i]
        after = series[i:]

        mean_before = np.mean(before)
        mean_after = np.mean(after)

        var_before = np.var(before, ddof=1) if len(before) > 1 else 0
        var_after = np.var(after, ddof=1) if len(after) > 1 else 0

        # Pooled variance
        pooled_var = ((len(before) - 1) * var_before + (len(after) - 1) * var_after) / (n - 2)

        # Add a tiny epsilon to variance to avoid division by zero
        pooled_var = max(pooled_var, 1e-10)

        # Welchs t-test like statistic
        diff = abs(mean_before - mean_after) / np.sqrt(pooled_var * (1/len(before) + 1/len(after)))

        if diff > max_diff and diff > threshold:
            max_diff = diff
            best_split = i

    return best_split
