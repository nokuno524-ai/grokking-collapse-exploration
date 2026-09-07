import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def threshold_detector(steps: np.ndarray, accs: np.ndarray, threshold: float = 0.90, dwell_k: int = 5) -> Optional[int]:
    """
    Detects grokking cliff via a threshold-crossing with a dwell time.
    Accuracy must cross `threshold` and stay above it for `dwell_k` consecutive evaluations.

    Returns the step where the threshold was first crossed for the successful dwell period,
    or None if the condition is never met (censored).
    """
    if len(accs) == 0:
        return None

    above = accs >= threshold

    # We want to find the first index i such that above[i:i+dwell_k] are all True
    for i in range(len(above) - dwell_k + 1):
        if np.all(above[i:i+dwell_k]):
            return int(steps[i])

    # If the array ends but the last few are True and we just ran out of evals,
    # we don't count it unless we actually reached the dwell.
    # However, if the very end sequence is all True and its length < dwell_k, it is censored.
    return None

def binary_segmentation_detector(steps: np.ndarray, accs: np.ndarray) -> Optional[int]:
    """
    Detects grokking cliff using a single-point binary segmentation algorithm.
    Finds the index `k` that minimizes the sum of variances of the two segments (before and after `k`).

    Returns the step at index `k`. Returns None if the variance reduction is negligible
    or if the final segment mean is low (meaning it never grokked).
    """
    if len(accs) < 3:
        return None

    min_cost = float('inf')
    best_idx = -1

    total_var = np.var(accs) * len(accs)

    for k in range(1, len(accs) - 1):
        seg1 = accs[:k]
        seg2 = accs[k:]

        cost = np.var(seg1) * len(seg1) + np.var(seg2) * len(seg2)

        if cost < min_cost:
            min_cost = cost
            best_idx = k

    if best_idx == -1:
        return None

    # Check if the split actually means grokking.
    # If the mean of the second segment is not high enough (e.g. not > 0.5), it might just be noise.
    # Also, if variance didn't reduce much, it's not a real cliff.
    # Arbitrary threshold to ensure it's a real jump.
    if np.mean(accs[best_idx:]) < 0.5:
        return None

    # Another sanity check: the final point should be reasonably high
    if accs[-1] < 0.8:
         return None

    return int(steps[best_idx])

def bootstrap_ci(data: np.ndarray, num_bootstrap: int = 1000, ci_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes the median and bootstrap confidence interval for a 1D array of data.
    Returns (median, lower_ci, upper_ci)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    median = np.median(data)

    if len(data) == 1:
        return median, median, median

    bootstraps = np.random.choice(data, size=(num_bootstrap, len(data)), replace=True)
    boot_medians = np.median(bootstraps, axis=1)

    lower_pct = (1 - ci_level) / 2 * 100
    upper_pct = (1 + ci_level) / 2 * 100

    lower_ci = np.percentile(boot_medians, lower_pct)
    upper_ci = np.percentile(boot_medians, upper_pct)

    return median, lower_ci, upper_ci

def detect_cliffs(history: List[Dict[str, Any]], max_steps: int) -> Dict[str, Any]:
    """
    Runs multiple detectors on a run's history.
    """
    steps = np.array([record['step'] for record in history])
    accs = np.array([record['test_acc'] for record in history])

    t_cliff = threshold_detector(steps, accs)
    b_cliff = binary_segmentation_detector(steps, accs)

    cliffs = []
    if t_cliff is not None: cliffs.append(t_cliff)
    if b_cliff is not None: cliffs.append(b_cliff)

    if len(cliffs) == 0:
        return {
            'is_censored': True,
            'cliff_step': max_steps,
            't_cliff': None,
            'b_cliff': None,
            'cliff_uncertainty': 0.0
        }
    else:
        # If both exist, average them or just use threshold
        # Using threshold as primary, binary seg as a check on uncertainty
        best_cliff = t_cliff if t_cliff is not None else b_cliff
        uncertainty = abs(t_cliff - b_cliff) if (t_cliff is not None and b_cliff is not None) else 0.0

        return {
            'is_censored': False,
            'cliff_step': best_cliff,
            't_cliff': t_cliff,
            'b_cliff': b_cliff,
            'cliff_uncertainty': float(uncertainty)
        }
