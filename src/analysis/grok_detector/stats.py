import numpy as np
from typing import Tuple, List, Dict

def kaplan_meier_estimator(durations: np.ndarray, event_observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Kaplan-Meier survival estimator.
    Args:
        durations: Array of times to event or censoring.
        event_observed: Boolean array, True if event (grokking) occurred, False if right-censored.
    Returns:
        times: Sorted unique event/censoring times.
        survival_prob: Kaplan-Meier survival probabilities.
        lower_ci: Lower 95% confidence interval (Greenwood's formula).
        upper_ci: Upper 95% confidence interval.
    """
    # Sort by duration
    idx = np.argsort(durations)
    d = durations[idx]
    e = event_observed[idx]

    unique_times = np.unique(d)
    n_times = len(unique_times)

    survival = np.ones(n_times)
    var = np.zeros(n_times)

    n_at_risk = len(d)
    current_surv = 1.0
    current_var_sum = 0.0

    for i, t in enumerate(unique_times):
        # Mask for current time
        mask = (d == t)
        events = np.sum(e[mask])
        censored = np.sum(~e[mask])

        if n_at_risk > 0:
            prob_survival = 1.0 - (events / n_at_risk)
            current_surv *= prob_survival

            # Greenwood's formula variance term
            if events > 0 and n_at_risk > events:
                current_var_sum += events / (n_at_risk * (n_at_risk - events))

        survival[i] = current_surv
        var[i] = (current_surv ** 2) * current_var_sum

        n_at_risk -= (events + censored)

    # 95% CI (1.96 standard errors)
    se = np.sqrt(np.maximum(var, 0))
    lower_ci = np.maximum(survival - 1.96 * se, 0.0)
    upper_ci = np.minimum(survival + 1.96 * se, 1.0)

    return unique_times, survival, lower_ci, upper_ci

def log_rank_test(durations_1: np.ndarray, event_observed_1: np.ndarray,
                  durations_2: np.ndarray, event_observed_2: np.ndarray) -> Tuple[float, float]:
    """
    Log-rank test for comparing two survival distributions.
    Returns (chi_square_stat, p_value).
    """
    import scipy.stats as stats

    all_durations = np.concatenate([durations_1, durations_2])
    all_events = np.concatenate([event_observed_1, event_observed_2])
    group = np.concatenate([np.zeros(len(durations_1)), np.ones(len(durations_2))])

    unique_times = np.unique(all_durations[all_events == 1])

    o1, o2 = 0, 0
    e1, e2 = 0, 0
    v1 = 0

    for t in unique_times:
        mask1 = (durations_1 >= t)
        mask2 = (durations_2 >= t)

        n1 = np.sum(mask1)
        n2 = np.sum(mask2)
        n = n1 + n2

        if n == 0:
            continue

        m1 = np.sum(event_observed_1[(durations_1 == t)])
        m2 = np.sum(event_observed_2[(durations_2 == t)])
        m = m1 + m2

        o1 += m1
        o2 += m2

        expected1 = m * (n1 / n)
        e1 += expected1
        e2 += m * (n2 / n)

        if n > 1:
            v_t = m * (n1 / n) * (n2 / n) * ((n - m) / (n - 1))
            v1 += v_t

    if v1 == 0:
        return 0.0, 1.0

    z = (o1 - e1) / np.sqrt(v1)
    chi2 = z ** 2
    p_val = stats.chi2.sf(chi2, df=1)

    return chi2, p_val

def permutation_test_median(durations_1: np.ndarray, events_1: np.ndarray,
                            durations_2: np.ndarray, events_2: np.ndarray,
                            n_permutations: int = 1000) -> float:
    """
    Permutation test comparing the median time-to-event of two groups.
    Only considers uncensored events (where event==True).
    Returns p-value.
    """
    t1 = durations_1[events_1 == 1]
    t2 = durations_2[events_2 == 1]

    if len(t1) == 0 or len(t2) == 0:
        return 1.0

    obs_diff = np.abs(np.median(t1) - np.median(t2))

    pooled = np.concatenate([t1, t2])
    n1 = len(t1)

    count = 0
    for _ in range(n_permutations):
        permuted = np.random.permutation(pooled)
        p1 = permuted[:n1]
        p2 = permuted[n1:]
        diff = np.abs(np.median(p1) - np.median(p2))
        if diff >= obs_diff:
            count += 1

    return count / n_permutations
