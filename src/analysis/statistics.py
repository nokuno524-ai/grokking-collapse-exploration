import numpy as np

def bootstrap_ci(data, stat_func=np.mean, n_boot=1000, ci=95):
    """Computes bootstrap confidence interval for a given statistic."""
    data = np.array(data)
    boot_stats = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(sample))

    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return float(lower), float(upper)

def check_significance(data1, data2, n_boot=1000):
    """
    Checks if the difference in means between data1 and data2 is statistically significant
    using bootstrap hypothesis testing (null hypothesis: means are equal).
    """
    d1 = np.array(data1)
    d2 = np.array(data2)

    mean_diff_observed = np.mean(d1) - np.mean(d2)

    combined = np.concatenate([d1, d2])
    combined_mean = np.mean(combined)

    # shift data to have same mean
    d1_shifted = d1 - np.mean(d1) + combined_mean
    d2_shifted = d2 - np.mean(d2) + combined_mean

    boot_diffs = []
    for _ in range(n_boot):
        samp1 = np.random.choice(d1_shifted, size=len(d1), replace=True)
        samp2 = np.random.choice(d2_shifted, size=len(d2), replace=True)
        boot_diffs.append(np.mean(samp1) - np.mean(samp2))

    boot_diffs = np.array(boot_diffs)
    # p-value: proportion of bootstrap differences at least as extreme as observed
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(mean_diff_observed))

    return float(p_value)
