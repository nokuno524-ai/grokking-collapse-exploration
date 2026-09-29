import numpy as np
from typing import Tuple, Sequence, List

def bootstrap_ci(data: Sequence[float], num_bootstraps: int = 10000, confidence_level: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        data: Sequence of numerical values.
        num_bootstraps: Number of bootstrap samples.
        confidence_level: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        raise ValueError("Cannot compute CI for empty data.")

    data_array = np.array(data)
    rng = np.random.RandomState(seed)
    n = len(data_array)

    # Generate bootstrap samples
    indices = rng.randint(0, n, size=(num_bootstraps, n))
    samples = data_array[indices]

    # Compute mean of each sample
    sample_means = np.mean(samples, axis=1)

    # Compute percentiles
    alpha = 1.0 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(sample_means, lower_percentile)
    upper_bound = np.percentile(sample_means, upper_percentile)

    return float(lower_bound), float(upper_bound)


def permutation_test(group1: Sequence[float], group2: Sequence[float], num_permutations: int = 10000, seed: int = 42) -> float:
    """
    Perform a permutation test for the difference in means between two groups.

    Args:
        group1: Sequence of numerical values for group 1.
        group2: Sequence of numerical values for group 2.
        num_permutations: Number of permutations.
        seed: Random seed for reproducibility.

    Returns:
        p-value for the two-sided test.
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    if len(g1) == 0 or len(g2) == 0:
        raise ValueError("Both groups must be non-empty.")

    observed_diff = np.abs(np.mean(g1) - np.mean(g2))

    combined = np.concatenate([g1, g2])
    n1 = len(g1)

    rng = np.random.RandomState(seed)

    count_extreme = 0

    for _ in range(num_permutations):
        shuffled = rng.permutation(combined)
        perm_g1 = shuffled[:n1]
        perm_g2 = shuffled[n1:]
        perm_diff = np.abs(np.mean(perm_g1) - np.mean(perm_g2))

        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = count_extreme / num_permutations
    return float(p_value)


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """
    Apply Holm-Bonferroni correction to a list of p-values.

    Args:
        p_values: Sequence of uncorrected p-values.

    Returns:
        List of corrected p-values.
    """
    if not p_values:
        return []

    p_values_arr = np.array(p_values)
    n = len(p_values_arr)

    # Sort indices
    sorted_indices = np.argsort(p_values_arr)
    sorted_p_values = p_values_arr[sorted_indices]

    # Apply correction
    corrected = np.empty(n)
    for i, p in enumerate(sorted_p_values):
        multiplier = n - i
        corr_p = p * multiplier
        corrected[sorted_indices[i]] = corr_p

    # Ensure monotonic increasing sequence and cap at 1.0
    for i in range(1, n):
        idx_prev = sorted_indices[i - 1]
        idx_curr = sorted_indices[i]
        corrected[idx_curr] = max(corrected[idx_prev], corrected[idx_curr])

    corrected = np.minimum(corrected, 1.0)

    return corrected.tolist()
