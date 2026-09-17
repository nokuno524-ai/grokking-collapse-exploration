import numpy as np
from typing import List, Dict, Any, Tuple

def compute_kaplan_meier(runs: List[Dict[str, Any]], max_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Kaplan-Meier estimator for grokking step.

    Args:
        runs: List of run dictionaries (from results.json).
        max_steps: Maximum number of steps in training.

    Returns:
        times: Array of steps where an event (grokking) occurred.
        survival_probs: Corresponding survival probabilities (probability of NOT grokking).
    """
    times = []
    events = [] # 1 if grokked, 0 if right-censored (did not grok)

    for r in runs:
        if r.get("grokked", False):
            grok_step = r.get("grokking_step")
            if grok_step is not None:
                times.append(grok_step)
                events.append(1)
        else:
            times.append(max_steps)
            events.append(0)

    if not times:
        return np.array([0, max_steps]), np.array([1.0, 1.0])

    times = np.array(times)
    events = np.array(events)

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    events = events[sort_idx]

    unique_times = np.unique(times)
    survival_probs = []

    n_at_risk = len(times)
    current_survival = 1.0

    # Include t=0, survival=1.0
    all_times = [0]
    survival_probs_list = [1.0]

    for t in unique_times:
        n_events = np.sum((times == t) & (events == 1))
        n_censored = np.sum((times == t) & (events == 0))

        if n_at_risk > 0:
            current_survival *= (1.0 - n_events / n_at_risk)

        all_times.append(t)
        survival_probs_list.append(current_survival)

        n_at_risk -= (n_events + n_censored)

    return np.array(all_times), np.array(survival_probs_list)

def bootstrap_ci(data: np.ndarray, num_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes bootstrap confidence interval for the mean.

    Args:
        data: 1D numpy array of data.
        num_bootstraps: Number of bootstrap resamples.
        ci: Confidence interval fraction (e.g. 0.95 for 95%).

    Returns:
        (mean, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    data = np.asarray(data)
    means = np.zeros(num_bootstraps)
    for i in range(num_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower_bound = np.percentile(means, alpha / 2.0 * 100)
    upper_bound = np.percentile(means, (1.0 - alpha / 2.0) * 100)

    return np.mean(data), lower_bound, upper_bound

def permutation_test(group1: np.ndarray, group2: np.ndarray, num_permutations: int = 10000) -> float:
    """
    Performs a non-parametric permutation test for difference in means.

    Args:
        group1: Data for group 1.
        group2: Data for group 2.
        num_permutations: Number of permutations.

    Returns:
        p_value: The two-sided p-value.
    """
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)

    if len(g1) == 0 or len(g2) == 0:
        return np.nan

    observed_diff = np.abs(np.mean(g1) - np.mean(g2))

    pooled = np.concatenate([g1, g2])
    n1 = len(g1)

    diffs = np.zeros(num_permutations)
    for i in range(num_permutations):
        permuted = np.random.permutation(pooled)
        p1 = permuted[:n1]
        p2 = permuted[n1:]
        diffs[i] = np.abs(np.mean(p1) - np.mean(p2))

    p_value = np.sum(diffs >= observed_diff) / num_permutations
    return p_value

def spearman_correlation_with_ci(x: np.ndarray, y: np.ndarray, num_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Spearman rank correlation with bootstrap confidence intervals.

    Args:
        x: Array of x values.
        y: Array of y values.
        num_bootstraps: Number of bootstraps for CI.
        ci: Confidence interval fraction.

    Returns:
        (correlation, lower_bound, upper_bound)
    """
    from scipy.stats import spearmanr

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2 or len(x) != len(y):
        return np.nan, np.nan, np.nan

    corr, _ = spearmanr(x, y)

    corrs = np.zeros(num_bootstraps)
    for i in range(num_bootstraps):
        indices = np.random.choice(len(x), size=len(x), replace=True)
        # Handle cases where bootstrap sample has uniform values causing warning/nan
        cx, cy = x[indices], y[indices]
        if np.std(cx) == 0 or np.std(cy) == 0:
            corrs[i] = 0.0
        else:
            corrs[i], _ = spearmanr(cx, cy)

    alpha = 1.0 - ci
    lower_bound = np.percentile(corrs, alpha / 2.0 * 100)
    upper_bound = np.percentile(corrs, (1.0 - alpha / 2.0) * 100)

    return corr, lower_bound, upper_bound
