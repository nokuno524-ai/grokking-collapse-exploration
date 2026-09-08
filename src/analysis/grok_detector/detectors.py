import numpy as np
from typing import List, Tuple, Optional, Callable

def threshold_detector(steps: np.ndarray, accs: np.ndarray, threshold: float = 0.95, dwell: int = 1) -> Optional[float]:
    """
    Find the first step where accuracy crosses the threshold and stays above it for 'dwell' consecutive points.
    """
    if len(steps) == 0:
        return None

    for i in range(len(steps) - dwell + 1):
        if np.all(accs[i:i+dwell] >= threshold):
            return float(steps[i])
    return None

def logistic_detector(steps: np.ndarray, accs: np.ndarray, threshold: float = 0.95) -> Optional[float]:
    """
    Fit a logistic curve L / (1 + exp(-k(x - x0))) + b to the accuracy curve.
    Return the step x where the fitted curve crosses the threshold.
    """
    from scipy.optimize import curve_fit

    if len(steps) < 4:
        return None

    def logistic(x, L, k, x0, b):
        z = np.clip(k * (x - x0), -500, 500) # prevent overflow
        return L / (1 + np.exp(-z)) + b

    try:
        # Initial guesses: L=1, k=0.01, x0=median step, b=0
        p0 = [1.0, 0.01, np.median(steps), 0.0]
        # Bounds: L in [0, 2], k in [1e-5, 10], x0 in [0, max(steps)*2], b in [-1, 1]
        bounds = ([0.0, 1e-5, 0.0, -1.0], [2.0, 10.0, max(steps)*2, 1.0])

        popt, _ = curve_fit(logistic, steps, accs, p0=p0, bounds=bounds, maxfev=10000)
        L, k, x0, b = popt

        # Solve for x: L / (1 + exp(-k(x - x0))) + b = threshold
        # L / (threshold - b) - 1 = exp(-k(x - x0))
        # -k(x - x0) = log(L / (threshold - b) - 1)
        # x = x0 - (1/k) * log(L / (threshold - b) - 1)

        val = L / (threshold - b) - 1
        if val <= 0:
            return None # Cannot reach threshold

        x_cross = x0 - (1.0 / k) * np.log(val)

        # We only accept if x_cross is roughly within our domain + some margin
        if x_cross < 0 or x_cross > max(steps) * 2:
            return None

        return float(x_cross)
    except Exception:
        return None

def binary_segmentation_detector(steps: np.ndarray, accs: np.ndarray) -> Optional[float]:
    """
    Find changepoint that maximizes variance reduction.
    """
    if len(steps) < 3:
        return None

    best_var_reduction = -1.0
    best_idx = -1

    total_var = np.var(accs) * len(accs)

    for i in range(1, len(steps) - 1):
        var_left = np.var(accs[:i]) * i
        var_right = np.var(accs[i:]) * (len(accs) - i)
        reduction = total_var - (var_left + var_right)

        if reduction > best_var_reduction:
            best_var_reduction = reduction
            best_idx = i

    if best_idx != -1:
        return float(steps[best_idx])
    return None

def derivative_maximum_detector(steps: np.ndarray, accs: np.ndarray, window: int = 3) -> Optional[float]:
    """
    Find the point of maximum smoothed derivative (the inflection point of the jump).
    """
    if len(steps) < window * 2:
        return None

    # smooth accs
    kernel = np.ones(window) / window
    smoothed_accs = np.convolve(accs, kernel, mode='valid')
    smoothed_steps = steps[window//2 : window//2 + len(smoothed_accs)]

    # compute diffs
    diffs = np.diff(smoothed_accs)
    step_diffs = np.diff(smoothed_steps)

    # avoid division by zero
    valid = step_diffs > 0
    if not np.any(valid):
         return None

    derivs = np.zeros_like(diffs)
    derivs[valid] = diffs[valid] / step_diffs[valid]

    max_idx = np.argmax(derivs)
    return float(smoothed_steps[max_idx])

def bootstrap_ci(steps: np.ndarray, accs: np.ndarray,
                 detector_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
                 n_bootstraps: int = 100,
                 ci: float = 0.95) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute confidence intervals on the detected changepoint using resampling with jitter.
    Returns (median_step, lower_ci, upper_ci)
    """
    base_val = detector_fn(steps, accs)
    if base_val is None:
        return None, None, None

    bootstrapped_vals = []
    n = len(steps)

    for _ in range(n_bootstraps):
        indices = np.random.choice(n, n, replace=True)
        # sort indices to maintain time ordering
        indices.sort()

        b_steps = steps[indices].astype(float)
        b_accs = accs[indices].copy()

        # Add tiny jitter to steps to avoid perfectly duplicate x-values which break curve_fit
        jitter = np.random.uniform(-0.1, 0.1, size=n)
        b_steps += jitter

        # Ensure strict monotonicity after jitter (simple fix: sort again)
        sort_idx = np.argsort(b_steps)
        b_steps = b_steps[sort_idx]
        b_accs = b_accs[sort_idx]

        val = detector_fn(b_steps, b_accs)
        if val is not None:
            bootstrapped_vals.append(val)

    if len(bootstrapped_vals) < n_bootstraps * 0.1: # if less than 10% succeeded
        return base_val, None, None

    alpha = (1.0 - ci) / 2.0
    lower = np.percentile(bootstrapped_vals, alpha * 100)
    upper = np.percentile(bootstrapped_vals, (1.0 - alpha) * 100)

    return base_val, float(lower), float(upper)
