import numpy as np
import scipy.stats as stats

def mean_ci(data, confidence=0.95):
    """
    Computes the mean and confidence interval for an array of data.
    Returns (mean, lower_bound, upper_bound).
    """
    a = 1.0 * np.array(data)
    if len(a) == 0:
        return float('nan'), float('nan'), float('nan')
    if len(a) == 1:
        return a[0], a[0], a[0]

    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def bootstrap_ci(data, num_bootstraps=10000, confidence=0.95):
    """
    Computes the mean and bootstrap confidence interval for an array of data.
    """
    a = np.array(data)
    if len(a) == 0:
        return float('nan'), float('nan'), float('nan')
    if len(a) == 1:
        return a[0], a[0], a[0]

    m = np.mean(a)
    bootstraps = np.random.choice(a, size=(num_bootstraps, len(a)), replace=True)
    means = np.mean(bootstraps, axis=1)

    alpha = 1.0 - confidence
    lower = np.percentile(means, alpha / 2.0 * 100)
    upper = np.percentile(means, (1.0 - alpha / 2.0) * 100)

    return m, lower, upper

def permutation_test(group_a, group_b, num_permutations=10000):
    """
    Performs a permutation test to compare the means of two groups.
    Returns the p-value for the hypothesis that the means are different.
    """
    a = np.array(group_a)
    b = np.array(group_b)

    if len(a) == 0 or len(b) == 0:
        return float('nan')

    obs_diff = abs(np.mean(a) - np.mean(b))

    combined = np.concatenate([a, b])
    n_a = len(a)

    count_extreme = 0

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = abs(np.mean(perm_a) - np.mean(perm_b))

        if perm_diff >= obs_diff:
            count_extreme += 1

    p_value = count_extreme / num_permutations
    return p_value

def analyze_grokking_incidence(registry, condition_a, condition_b):
    """
    Extracts grokking (boolean) for two conditions and runs a permutation test.
    """
    runs_a = [r["grokked"] for r in registry if r.get("condition_name") == condition_a]
    runs_b = [r["grokked"] for r in registry if r.get("condition_name") == condition_b]

    p_val = permutation_test(runs_a, runs_b)

    m_a, _, _ = mean_ci(runs_a)
    m_b, _, _ = mean_ci(runs_b)

    return {
        "condition_a_mean": m_a,
        "condition_b_mean": m_b,
        "n_a": len(runs_a),
        "n_b": len(runs_b),
        "p_value": p_val
    }
