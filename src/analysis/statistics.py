import numpy as np
import scipy.stats as stats
from typing import Tuple, List

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d for two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd

def compute_bootstrap_ci(data: np.ndarray, n_resamples: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    resamples = rng.choice(data, size=(n_resamples, len(data)), replace=True)
    means = np.mean(resamples, axis=1)
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)

def t_test_independent(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Perform independent t-test."""
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    return float(t_stat), float(p_val)
