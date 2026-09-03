import numpy as np
from typing import Tuple

def kaplan_meier(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Kaplan-Meier estimator for survival probability.

    Args:
        times: Array of times at which events or censoring occurred.
        events: Boolean array indicating if the event (grokking) occurred (1/True)
                or if the observation was censored (0/False).

    Returns:
        unique_times: Sorted unique event times.
        survival_probs: Estimated probability of survival (not grokking) past each time.
    """
    if len(times) != len(events):
        raise ValueError("times and events must have the same length")

    if len(times) == 0:
        return np.array([]), np.array([])

    # Sort by time
    sort_idx = np.argsort(times)
    sorted_times = times[sort_idx]
    sorted_events = events[sort_idx]

    unique_times = np.unique(sorted_times)
    survival_probs = np.ones_like(unique_times, dtype=float)

    n_at_risk = len(sorted_times)
    current_survival = 1.0

    for i, t in enumerate(unique_times):
        # Number of events (groks) at time t
        n_events = np.sum((sorted_times == t) & (sorted_events == 1))

        # Number of censored observations at time t
        n_censored = np.sum((sorted_times == t) & (sorted_events == 0))

        if n_at_risk > 0:
            current_survival *= (1.0 - n_events / n_at_risk)

        survival_probs[i] = current_survival
        n_at_risk -= (n_events + n_censored)

    return unique_times, survival_probs
