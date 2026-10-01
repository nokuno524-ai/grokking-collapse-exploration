import numpy as np
from typing import List, Tuple, Dict, Any

def leave_one_out_stability(steps_list: List[List[int]], accs_list: List[List[float]], detector_fn) -> Dict[str, Any]:
    """
    Performs leave-one-seed-out stability analysis of cliff locations.

    Args:
        steps_list: A list of step lists (one for each seed).
        accs_list: A list of accuracy lists (one for each seed).
        detector_fn: A function that takes (steps, accs) and returns a cliff step (or None).

    Returns:
        A dictionary containing stability metrics.
    """
    n_seeds = len(steps_list)
    if n_seeds < 2:
        return {'error': 'Need at least 2 seeds for stability analysis'}

    # Detect cliffs on all seeds
    all_cliffs = []
    valid_indices = []
    for i in range(n_seeds):
        c = detector_fn(steps_list[i], accs_list[i])
        if c is not None:
            all_cliffs.append(c)
            valid_indices.append(i)

    if not all_cliffs:
        return {'error': 'No cliffs detected in any seed'}

    if len(all_cliffs) == 1:
        return {
            'all_cliffs': all_cliffs,
            'median_all': all_cliffs[0],
            'loo_medians': [],
            'max_deviation': 0.0
        }

    median_all = np.median(all_cliffs)

    loo_medians = []
    for i in range(len(all_cliffs)):
        loo_sample = all_cliffs[:i] + all_cliffs[i+1:]
        loo_medians.append(np.median(loo_sample))

    max_dev = max([abs(m - median_all) for m in loo_medians])

    return {
        'all_cliffs': all_cliffs,
        'median_all': float(median_all),
        'loo_medians': [float(m) for m in loo_medians],
        'max_deviation': float(max_dev)
    }

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Cliff's Delta effect size for two independent samples.
    Cliff's Delta = (P(x > y) - P(x < y))
    Ranges from -1 to 1.
    """
    nx = len(x)
    ny = len(y)

    if nx == 0 or ny == 0:
        return float('nan')

    # Compare each x to each y
    diff = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                diff += 1
            elif xi < yi:
                diff -= 1

    return diff / (nx * ny)


def effect_size_stability(x: np.ndarray, y: np.ndarray, effect_size_fn) -> Dict[str, Any]:
    """
    Performs leave-one-out stability analysis on an effect size comparison between two groups.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 2 or len(y) < 2:
        return {'error': 'Need at least 2 samples per group for stability analysis'}

    full_estimate = effect_size_fn(x, y)

    loo_estimates = []
    # Leave one out from x
    for i in range(len(x)):
        x_loo = np.concatenate([x[:i], x[i+1:]])
        loo_estimates.append(effect_size_fn(x_loo, y))

    # Leave one out from y
    for i in range(len(y)):
        y_loo = np.concatenate([y[:i], y[i+1:]])
        loo_estimates.append(effect_size_fn(x, y_loo))

    max_dev = max([abs(est - full_estimate) for est in loo_estimates if not np.isnan(est)]) if loo_estimates else 0.0

    return {
        'full_estimate': float(full_estimate),
        'loo_estimates': [float(e) for e in loo_estimates],
        'max_deviation': float(max_dev)
    }
