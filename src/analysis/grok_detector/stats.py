import numpy as np
import scipy.stats as stats
from typing import Tuple, List, Optional

def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate the Wilson score interval for a binomial proportion.
    Returns (proportion, lower_bound, upper_bound)
    """
    if n == 0:
        return 0.0, 0.0, 0.0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return p, float(lower), float(upper)

def kaplan_meier(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Kaplan-Meier survival curve.
    times: array of event times or censoring times
    events: array of booleans (1 if event occurred, 0 if censored)
    Returns (unique_times, survival_probabilities)
    """
    # Sort times and events together
    idx = np.argsort(times)
    t = times[idx]
    e = events[idx]

    unique_t = np.unique(t)
    survival = np.zeros_like(unique_t, dtype=float)

    n_at_risk = len(t)
    current_survival = 1.0

    for i, t_i in enumerate(unique_t):
        # find all events happening at this exact time
        mask = (t == t_i)
        n_events = np.sum(e[mask])
        n_censored = np.sum(~e[mask])

        if n_at_risk > 0:
            current_survival *= (1.0 - n_events / n_at_risk)

        survival[i] = current_survival
        n_at_risk -= (n_events + n_censored)

    return unique_t, survival

def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_sd)

def bootstrap_effect_size(group1: np.ndarray, group2: np.ndarray,
                          n_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute Cohen's d with bootstrap confidence intervals.
    Returns (d, lower_ci, upper_ci)
    """
    d = cohen_d(group1, group2)

    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return d, d, d

    bootstrapped_d = []
    for _ in range(n_bootstraps):
        b1 = np.random.choice(group1, size=n1, replace=True)
        b2 = np.random.choice(group2, size=n2, replace=True)
        bootstrapped_d.append(cohen_d(b1, b2))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrapped_d, alpha * 100))
    upper = float(np.percentile(bootstrapped_d, (1.0 - alpha) * 100))

    return d, lower, upper

def holm_mann_whitney(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Mann-Whitney U test p-value.
    Handles NaN/ValueError gracefully when arrays have identical values.
    Returns p-value.
    """
    try:
        # if all values are identical or arrays too small, p=1.0
        if len(group1) == 0 or len(group2) == 0:
            return 1.0
        if np.all(group1 == group1[0]) and np.all(group2 == group2[0]) and group1[0] == group2[0]:
            return 1.0

        _, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        if np.isnan(p_val):
            return 1.0
        return float(p_val)
    except ValueError:
        return 1.0
