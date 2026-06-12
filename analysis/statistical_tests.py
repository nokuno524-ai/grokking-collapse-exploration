import numpy as np
from scipy import stats
from typing import Tuple, Optional

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Computes Cohen's d effect size between two groups.
    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)

    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std

def bootstrap_ci(data: np.ndarray, confidence: float = 0.95, n_resamples: int = 9999) -> Tuple[float, float]:
    """
    Computes the BCa (bias-corrected and accelerated) bootstrap confidence interval for the mean.
    Returns (lower_bound, upper_bound).
    """
    if len(data) < 2:
        return np.mean(data) if len(data) > 0 else 0.0, np.mean(data) if len(data) > 0 else 0.0

    res = stats.bootstrap((data,), np.mean, confidence_level=confidence, method='BCa', n_resamples=n_resamples, axis=-1)
    return res.confidence_interval.low, res.confidence_interval.high

def mwu_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """
    Performs the Mann-Whitney U test between two groups.
    Returns (statistic, p_value).
    """
    if len(group1) == 0 or len(group2) == 0:
        return 0.0, 1.0

    res = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return res.statistic, res.pvalue

if __name__ == "__main__":
    # Simple self-test
    g1 = np.random.normal(loc=0.0, scale=1.0, size=50)
    g2 = np.random.normal(loc=1.0, scale=1.0, size=50)

    d = cohens_d(g1, g2)
    ci = bootstrap_ci(g1)
    stat, pval = mwu_test(g1, g2)

    print(f"Cohen's d: {d:.3f}")
    print(f"Group 1 Bootstrap CI: ({ci[0]:.3f}, {ci[1]:.3f})")
    print(f"MWU p-value: {pval:.3e}")
