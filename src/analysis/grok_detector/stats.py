import numpy as np
from typing import Tuple, List, Dict, Any, Optional

def kaplan_meier_median(times: np.ndarray, events: np.ndarray) -> Optional[float]:
    """
    Computes the Kaplan-Meier median time-to-event for censored data.

    Args:
        times: Array of times to event or censorship.
        events: Array of boolean indicators (1 if event occurred, 0 if censored).

    Returns:
        The estimated median time, or None if the survival probability doesn't drop below 0.5.
    """
    if len(times) == 0:
        return None

    order = np.argsort(times)
    t = times[order]
    e = events[order]

    survival_prob = 1.0
    n_at_risk = len(t)

    # Track survival curve: list of (time, probability)
    curve = [(0.0, 1.0)]

    for i in range(len(t)):
        if e[i] == 1:
            survival_prob *= (1.0 - 1.0 / n_at_risk)
        curve.append((t[i], survival_prob))

        if survival_prob <= 0.5:
            return float(t[i])

        n_at_risk -= 1

    return None

def kaplan_meier_ci(times: np.ndarray, events: np.ndarray, ci_level: float = 95.0, n_bootstraps: int = 1000) -> Tuple[float, float]:
    """
    Bootstraps the Kaplan-Meier median to get a confidence interval.

    Args:
        times: Array of times to event or censorship.
        events: Array of boolean indicators.
        ci_level: Confidence level in percent.
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        (lower_bound, upper_bound)
    """
    n = len(times)
    if n == 0:
        return np.nan, np.nan

    medians = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        resampled_t = times[indices]
        resampled_e = events[indices]
        med = kaplan_meier_median(resampled_t, resampled_e)
        if med is not None:
            medians.append(med)

    if not medians:
        return np.nan, np.nan

    alpha = (100.0 - ci_level) / 2.0
    lower = np.percentile(medians, alpha)
    upper = np.percentile(medians, 100.0 - alpha)

    return float(lower), float(upper)

def aggregate_seeds(results_list: List[Dict[str, Any]], max_step: float) -> Dict[str, Any]:
    """
    Aggregates grokking steps across multiple seeds, computing Kaplan-Meier statistics
    to handle non-grokking (censored) seeds.

    Args:
        results_list: List of result dicts, each with a 'grokking_step' (None if not grokked).
        max_step: The maximum step reached in training (used as censorship time).

    Returns:
        Dictionary with aggregated stats.
    """
    times = []
    events = []

    for r in results_list:
        step = r.get("grokking_step")
        if step is not None:
            times.append(step)
            events.append(1)
        else:
            times.append(max_step)
            events.append(0)

    times = np.array(times, dtype=float)
    events = np.array(events, dtype=int)

    median = kaplan_meier_median(times, events)
    ci_lower, ci_upper = kaplan_meier_ci(times, events)

    # Calculate simple rate
    grok_rate = np.mean(events)

    return {
        "median": median,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "grok_rate": float(grok_rate),
        "n_seeds": len(results_list),
        "n_grokked": int(np.sum(events)),
        "times": times.tolist(),
        "events": events.tolist()
    }

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Computes Cohen's d effect size between two groups.
    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)

def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Computes Cliff's delta effect size between two groups (non-parametric).
    delta = (P(x1 > x2) - P(x1 < x2))
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan

    gt = sum(x1 > x2 for x1 in group1 for x2 in group2)
    lt = sum(x1 < x2 for x1 in group1 for x2 in group2)

    return float((gt - lt) / (n1 * n2))
