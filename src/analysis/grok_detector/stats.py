"""
Statistical tools for robust cliff detection and survival analysis of grokking.
"""

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any, Optional

def detect_grokking_step(
    steps: np.ndarray,
    accuracies: np.ndarray,
    threshold: float = 0.95,
    stability_window: int = 5
) -> Optional[int]:
    """
    Detect the first step where accuracy crosses the threshold and stays above it
    for `stability_window` consecutive evaluation steps.

    Args:
        steps: Array of evaluation steps.
        accuracies: Array of test accuracies corresponding to steps.
        threshold: The accuracy threshold to cross.
        stability_window: Number of consecutive steps accuracy must remain >= threshold.

    Returns:
        The step number where grokking is achieved, or None if not grokked.
    """
    if len(steps) != len(accuracies) or len(steps) == 0:
        return None

    # Floating point safe comparison
    epsilon = 1e-6
    above_threshold = accuracies >= (threshold - epsilon)

    for i in range(len(above_threshold) - stability_window + 1):
        if np.all(above_threshold[i : i + stability_window]):
            return steps[i]

    # If the window extends beyond the end of the array but all remaining are above threshold
    if len(above_threshold) > 0 and np.all(above_threshold[-min(stability_window, len(above_threshold)):]):
        # We might consider it grokked if it reached the end, but strictly we need K evals.
        # Let's be strict: if it didn't have enough evals, we don't count it.
        # Except if the array itself is smaller than stability window
        if len(above_threshold) >= stability_window:
            pass # Already covered by loop

    return None


def bootstrap_median_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute the median and bootstrapped confidence intervals.
    Filters out NaNs prior to computation.

    Args:
        data: Array of values (e.g., grokking steps).
        confidence: Confidence level.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Tuple of (median, lower_ci, upper_ci). Returns (NaN, NaN, NaN) if data is empty or all NaNs.
    """
    clean_data = np.asarray(data)
    clean_data = clean_data[~np.isnan(clean_data)]

    if len(clean_data) == 0:
        return np.nan, np.nan, np.nan

    median = np.median(clean_data)

    # If all values are identical, CI is just that value
    if np.all(clean_data == clean_data[0]):
        return median, clean_data[0], clean_data[0]

    rng = np.random.RandomState(seed)

    # Bootstrap
    medians = []
    n = len(clean_data)
    for _ in range(n_resamples):
        sample = rng.choice(clean_data, size=n, replace=True)
        medians.append(np.median(sample))

    medians = np.array(medians)
    alpha = 1.0 - confidence
    lower_ci = np.percentile(medians, alpha / 2.0 * 100)
    upper_ci = np.percentile(medians, (1.0 - alpha / 2.0) * 100)

    return float(median), float(lower_ci), float(upper_ci)


def kaplan_meier_survival(
    times: np.ndarray,
    censored: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Kaplan-Meier survival curve for grokking.
    Here, "survival" means the probability of NOT grokking by time t.

    Args:
        times: Array of times (e.g. training steps) of events or censoring.
        censored: Boolean array (True if run did NOT grok, False if it DID grok).

    Returns:
        Tuple of (unique_times, survival_probs, lower_ci, upper_ci) using Greenwood's formula for 95% CIs.
    """
    if len(times) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Sort by time
    idx = np.argsort(times)
    times_sorted = times[idx]
    censored_sorted = censored[idx]

    unique_times = np.unique(times_sorted)

    n_at_risk = len(times)
    survival = 1.0

    survival_probs = []
    variances = [] # For Greenwood's formula
    cumulative_variance = 0.0

    for t in unique_times:
        # Number of events (grokkings) at time t
        events_at_t = np.sum((times_sorted == t) & (~censored_sorted))
        # Number of censored at time t
        censored_at_t = np.sum((times_sorted == t) & censored_sorted)

        if n_at_risk > 0:
            survival_t = survival * (1.0 - events_at_t / n_at_risk)
            # Greenwood's variance term
            if n_at_risk > events_at_t:
                cumulative_variance += events_at_t / (n_at_risk * (n_at_risk - events_at_t))
            elif events_at_t > 0:
                # If everyone at risk had an event, variance is infinite/undefined, but probability is 0
                cumulative_variance = np.inf
        else:
            survival_t = 0.0
            cumulative_variance = np.inf

        survival = survival_t
        survival_probs.append(survival)

        # log-log transformation for CI
        if survival > 0 and survival < 1 and cumulative_variance < np.inf:
            var_log_log = cumulative_variance / (np.log(survival)**2)
            variances.append(var_log_log)
        else:
            variances.append(0.0) # Dummy variance when survival is 1 or 0

        n_at_risk -= (events_at_t + censored_at_t)

    survival_probs = np.array(survival_probs)
    variances = np.array(variances)

    # 95% CI (z = 1.96)
    z = 1.96
    lower_ci = []
    upper_ci = []

    for s, v in zip(survival_probs, variances):
        if s == 1.0:
            lower_ci.append(1.0)
            upper_ci.append(1.0)
        elif s == 0.0:
            lower_ci.append(0.0)
            upper_ci.append(0.0)
        elif v == 0.0:
             lower_ci.append(s)
             upper_ci.append(s)
        else:
            # log-log CI
            w = np.exp(z * np.sqrt(v))
            l = s ** w
            u = s ** (1/w)
            lower_ci.append(l)
            upper_ci.append(u)

    return unique_times, survival_probs, np.array(lower_ci), np.array(upper_ci)


def log_rank_test_multi(
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]],
    method: str = 'bonferroni'
) -> Dict[Tuple[str, str], float]:
    """
    Perform pairwise log-rank tests (or Mann-Whitney U on time-to-event) across multiple groups
    and apply multiple comparison correction.

    Since true Log-Rank requires more complex implementation without `lifelines`,
    we implement a permutation-based approach or Mann-Whitney U for time-to-event with tied censoring.
    Here we'll use Scipy's log-rank test approximation or Mann-Whitney.
    Because exact log rank involves variance computation over combined pools, we will implement
    a simplified log-rank via chi-square on observed vs expected.

    Args:
        groups: Dictionary mapping condition_name to (times, censored) arrays.
        method: Multiple comparison correction method ('bonferroni').

    Returns:
        Dictionary of (group1, group2) -> corrected p-value.
    """
    group_names = list(groups.keys())
    p_values = {}

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name1, name2 = group_names[i], group_names[j]
            times1, cens1 = groups[name1]
            times2, cens2 = groups[name2]

            # Simplified exact log-rank test for two groups
            p_val = _two_sample_log_rank(times1, cens1, times2, cens2)
            p_values[(name1, name2)] = p_val

    # Apply Bonferroni correction
    if method == 'bonferroni':
        num_tests = len(p_values)
        for k in p_values:
            p_values[k] = min(1.0, p_values[k] * num_tests)

    return p_values


def _two_sample_log_rank(
    times1: np.ndarray, cens1: np.ndarray,
    times2: np.ndarray, cens2: np.ndarray
) -> float:
    """Helper for two-sample log rank test."""
    times = np.concatenate([times1, times2])
    cens = np.concatenate([cens1, cens2])
    group_indicator = np.concatenate([np.zeros(len(times1)), np.ones(len(times2))])

    unique_times = np.unique(times)

    obs1 = 0
    exp1 = 0
    obs2 = 0
    exp2 = 0
    var = 0.0

    for t in unique_times:
        mask = (times == t)
        n1 = np.sum((times >= t) & (group_indicator == 0))
        n2 = np.sum((times >= t) & (group_indicator == 1))
        n = n1 + n2

        if n == 0:
            continue

        m1 = np.sum(mask & (group_indicator == 0) & (~cens))
        m2 = np.sum(mask & (group_indicator == 1) & (~cens))
        m = m1 + m2

        if m == 0:
            continue

        obs1 += m1
        exp1 += m * (n1 / n)
        obs2 += m2
        exp2 += m * (n2 / n)

        if n > 1:
            var += (m * (n - m) * n1 * n2) / (n**2 * (n - 1))

    if var == 0:
        return 1.0

    chi2 = ((obs1 - exp1)**2) / var
    # 1 degree of freedom
    p_value = stats.chi2.sf(chi2, 1)

    # Handle NaN explicitly from scipy stats occasionally
    if np.isnan(p_value):
        return 1.0
    return float(p_value)
