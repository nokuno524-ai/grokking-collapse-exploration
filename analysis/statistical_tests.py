import numpy as np
import scipy.stats as stats

def compute_welch_ttest(data1, data2):
    """
    Computes Welch's t-test for two independent samples.
    """
    return stats.ttest_ind(data1, data2, equal_var=False)

def compute_ks_test(data1, data2):
    """
    Computes the Kolmogorov-Smirnov test for two distributions.
    """
    return stats.ks_2samp(data1, data2)

def compute_bootstrap_ci(data, confidence_level=0.95, n_resamples=9999):
    """
    Computes bootstrap confidence intervals for the mean of the data.
    Handles degenerate variance cases (where all data points are identical).
    """
    data = np.array(data)
    if len(data) < 2:
        return (data[0], data[0]) if len(data) == 1 else (np.nan, np.nan)

    # Check for degenerate variance
    if np.var(data) == 0:
        return (data[0], data[0])

    res = stats.bootstrap((data,), np.mean, confidence_level=confidence_level, n_resamples=n_resamples, method='BCa')
    return (res.confidence_interval.low, res.confidence_interval.high)

def compute_cohens_d(data1, data2):
    """
    Computes Cohen's d effect size for two groups.
    """
    n1, n2 = len(data1), len(data2)

    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)

    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return 0.0

    return (np.mean(data1) - np.mean(data2)) / pooled_sd
