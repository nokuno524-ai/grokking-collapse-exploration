import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from scipy import stats

def hedges_g(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes Hedge's g effect size between two groups, and its 95% Confidence Interval.
    Returns (g, ci_lower, ci_upper)
    """
    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        return float('nan'), float('nan'), float('nan')

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (mean1 - mean2) / s_pooled

    # Hedge's g correction for small sample sizes
    # g ≈ d * (1 - 3 / (4 * (n1 + n2) - 9))
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    g = d * correction

    # Variance of Hedge's g
    v_d = (n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2))
    v_g = v_d * (correction**2)

    # 95% CI (z ≈ 1.96)
    ci_half_width = 1.96 * np.sqrt(v_g)

    return g, g - ci_half_width, g + ci_half_width

def permutation_test(group1: np.ndarray, group2: np.ndarray, n_permutations: int = 10000, seed: int = 42) -> float:
    """
    Computes p-value using a permutation test on the difference of means.
    """
    np.random.seed(seed)

    n1 = len(group1)
    n2 = len(group2)
    if n1 == 0 or n2 == 0:
         return float('nan')

    obs_diff = abs(np.mean(group1) - np.mean(group2))

    pooled = np.concatenate([group1, group2])
    n_total = n1 + n2

    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        perm_g1 = pooled[:n1]
        perm_g2 = pooled[n1:]
        perm_diff = abs(np.mean(perm_g1) - np.mean(perm_g2))

        if perm_diff >= obs_diff:
            count_extreme += 1

    return count_extreme / n_permutations

def kaplan_meier_estimator(durations: np.ndarray, event_observed: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Kaplan-Meier estimator for survival analysis.
    Treats non-grokking (event_observed=False) as right-censored.

    Args:
        durations: Array of times to event (grokking step or max steps)
        event_observed: Boolean array where True means event occurred.

    Returns:
        Dict containing timeline, survival probabilities.
    """
    # Sort by duration
    order = np.argsort(durations)
    durations = durations[order]
    event_observed = event_observed[order]

    unique_times = np.unique(durations)
    survival_probs = []

    n_at_risk = len(durations)
    current_surv = 1.0

    timeline = []

    for t in unique_times:
        timeline.append(t)

        # Events at time t
        at_t = (durations == t)
        events_t = np.sum(event_observed[at_t])
        censored_t = np.sum(~event_observed[at_t])

        if n_at_risk > 0:
            current_surv *= (1.0 - events_t / n_at_risk)
        survival_probs.append(current_surv)

        n_at_risk -= (events_t + censored_t)

    return {
        "timeline": np.array([0] + timeline),
        "survival_probability": np.array([1.0] + survival_probs)
    }
