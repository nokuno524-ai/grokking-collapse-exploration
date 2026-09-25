"""
Statistical testing and rigorous metrics computation.
"""

import numpy as np
from scipy import stats

def compute_bootstrap_ci(data, num_bootstraps=1000, ci_level=0.95, seed=42):
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        data: List or array of numerical data.
        num_bootstraps: Number of bootstrap samples.
        ci_level: Confidence interval level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    data = np.array([x for x in data if x is not None and not np.isnan(x)])
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    if len(data) == 1:
        return data[0], data[0], data[0]

    rng = np.random.default_rng(seed)

    # Generate bootstrap samples
    bootstrap_samples = rng.choice(data, size=(num_bootstraps, len(data)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)

    # Calculate percentiles
    alpha = (1 - ci_level) / 2
    lower_bound = np.percentile(bootstrap_means, alpha * 100)
    upper_bound = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return np.mean(data), lower_bound, upper_bound

def run_mannwhitneyu(group1, group2, alternative='two-sided'):
    """
    Run Mann-Whitney U test between two groups.

    Args:
        group1: List or array of data for group 1.
        group2: List or array of data for group 2.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        dict: Test results including statistic and p-value.
    """
    g1 = np.array([x for x in group1 if x is not None and not np.isnan(x)])
    g2 = np.array([x for x in group2 if x is not None and not np.isnan(x)])

    if len(g1) == 0 or len(g2) == 0:
        return {'statistic': np.nan, 'pvalue': np.nan}

    try:
        stat, pval = stats.mannwhitneyu(g1, g2, alternative=alternative)
        return {'statistic': float(stat), 'pvalue': float(pval)}
    except ValueError:
        # E.g., all numbers are identical
        return {'statistic': np.nan, 'pvalue': np.nan}
