import numpy as np
from scipy.optimize import curve_fit
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import scipy.stats

def sigmoid(x: np.ndarray, bottom: float, top: float, k: float, x0: float) -> np.ndarray:
    """Logistic function for cliff fitting.
    y = bottom + (top - bottom) / (1 + exp(-k(x - x0)))
    """
    z = -k * (x - x0)
    z = np.clip(z, -500, 500)
    return bottom + (top - bottom) / (1.0 + np.exp(z))

def extract_cliff_stats(steps: np.ndarray, acc: np.ndarray) -> Dict[str, Any]:
    """Fits logistic curve to accuracy vs steps to extract cliff stats.

    Args:
        steps: Array of training steps.
        acc: Array of validation/test accuracies.

    Returns:
        Dict with:
            grokking_step: Step where accuracy crosses 50% of the transition (x0).
            cliff_width: Steps from 10% to 90% transition (2 * ln(9) / k).
            asymptotic_acc: The fitted upper asymptote (top).
            r2: Goodness of fit.
    """
    bottom = np.min(acc)
    top = np.max(acc)

    if top - bottom < 0.1:
        return {'grokking_step': np.nan, 'cliff_width': np.nan, 'asymptotic_acc': float(top), 'r2': 0.0}

    mid = (bottom + top) / 2
    x0_idx = np.argmax(acc > mid)
    x0 = steps[x0_idx] if np.any(acc > mid) else steps[-1]

    step_range = steps[-1] - steps[0]
    if step_range == 0:
        return {'grokking_step': np.nan, 'cliff_width': np.nan, 'asymptotic_acc': float(top), 'r2': 0.0}

    k_guess = 10.0 / step_range

    try:
        popt, _ = curve_fit(
            sigmoid, steps, acc,
            p0=[bottom, top, k_guess, x0],
            bounds=([0, 0, 0, 0], [1.0, 1.0, np.inf, np.inf]),
            maxfev=10000
        )

        width = 2 * np.log(9) / popt[2] if popt[2] > 0 else np.nan
        pred = sigmoid(steps, *popt)
        ss_res = np.sum((acc - pred)**2)
        ss_tot = np.sum((acc - np.mean(acc))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'grokking_step': float(popt[3]),
            'cliff_width': float(width),
            'asymptotic_acc': float(popt[1]),
            'r2': float(r2)
        }
    except Exception as e:
        return {'grokking_step': np.nan, 'cliff_width': np.nan, 'asymptotic_acc': float(top), 'r2': 0.0}

def permutation_test(val1: np.ndarray, val2: np.ndarray, n_permutations: int = 10000) -> float:
    """Permutation test for difference in means (val2 > val1 or val2 < val1 depending on direction).
    Tests the null hypothesis that val1 and val2 come from the same distribution.

    Returns the two-tailed p-value.
    """
    val1 = np.asarray(val1)[~np.isnan(val1)]
    val2 = np.asarray(val2)[~np.isnan(val2)]

    if len(val1) == 0 or len(val2) == 0:
        return np.nan

    obs_diff = np.abs(np.mean(val1) - np.mean(val2))
    if obs_diff == 0:
        return 1.0

    combined = np.concatenate([val1, val2])
    n1 = len(val1)

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        p_val1 = combined[:n1]
        p_val2 = combined[n1:]
        p_diff = np.abs(np.mean(p_val1) - np.mean(p_val2))
        if p_diff >= obs_diff:
            count += 1

    return count / n_permutations

def cohen_d(val1: np.ndarray, val2: np.ndarray) -> float:
    """Calculate Cohen's d for effect size."""
    val1 = np.asarray(val1)[~np.isnan(val1)]
    val2 = np.asarray(val2)[~np.isnan(val2)]

    if len(val1) < 2 or len(val2) < 2:
        return np.nan

    n1, n2 = len(val1), len(val2)
    var1, var2 = np.var(val1, ddof=1), np.var(val2, ddof=1)

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled_var == 0:
        return 0.0

    return (np.mean(val2) - np.mean(val1)) / np.sqrt(pooled_var)

def compute_ci(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval using t-distribution."""
    data = np.asarray(data)[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)

    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m + h

def trend_test(arrays: List[np.ndarray]) -> float:
    """Spearman rank correlation across ordered groups to test for a monotonic trend."""
    x = []
    y = []
    for i, arr in enumerate(arrays):
        arr = np.asarray(arr)[~np.isnan(arr)]
        for val in arr:
            x.append(i)
            y.append(val)

    if len(x) < 3:
        return np.nan

    corr, p_value = scipy.stats.spearmanr(x, y)
    return p_value
