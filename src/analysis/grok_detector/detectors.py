import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional, Callable, Dict, Any, List

def piecewise_constant_detector(steps: np.ndarray, metric: np.ndarray) -> Optional[float]:
    """
    Detects the grokking step using a piecewise-constant fit (binary segmentation).
    Finds the split index that minimizes the sum of squared errors for two constant segments.

    Args:
        steps: Array of evaluation steps.
        metric: Array of metric values (e.g., test accuracy).

    Returns:
        The step at which the changepoint occurs, or None if invalid.
    """
    if len(steps) < 3:
        return None

    best_split = None
    min_sse = float('inf')

    for i in range(1, len(steps) - 1):
        left_mean = np.mean(metric[:i])
        right_mean = np.mean(metric[i:])

        sse_left = np.sum((metric[:i] - left_mean)**2)
        sse_right = np.sum((metric[i:] - right_mean)**2)
        sse = sse_left + sse_right

        if sse < min_sse:
            min_sse = sse
            best_split = i

    if best_split is not None:
        return float(steps[best_split])
    return None

def logistic_function(x, L, k, x0, b):
    # np.clip to prevent overflow
    return L / (1 + np.exp(-np.clip(k * (x - x0), -500, 500))) + b

def logistic_detector(steps: np.ndarray, metric: np.ndarray) -> Optional[float]:
    """
    Detects the grokking step by fitting a logistic curve and finding the max slope (x0).

    Args:
        steps: Array of evaluation steps.
        metric: Array of metric values.

    Returns:
        The step corresponding to the maximum slope, or None if fit fails.
    """
    if len(steps) < 4:
        return None

    # Initial guesses: L=1, k=0.01, x0=mean(steps), b=0
    p0 = [1.0, 0.01, np.mean(steps), 0.0]

    try:
        popt, _ = curve_fit(logistic_function, steps, metric, p0=p0, maxfev=10000)
        x0 = popt[2]
        # Return x0 clamped to the step range
        if steps[0] <= x0 <= steps[-1]:
            return float(x0)
        else:
            return None
    except (RuntimeError, ValueError):
        return None

def threshold_detector(steps: np.ndarray, metric: np.ndarray, threshold: float = 0.9) -> Optional[float]:
    """
    Detects the first step where the metric exceeds a given threshold.

    Args:
        steps: Array of evaluation steps.
        metric: Array of metric values.
        threshold: The threshold to cross.

    Returns:
        The first step crossing the threshold, or None if never crossed.
    """
    crossed = metric >= threshold
    if not np.any(crossed):
        return None
    return float(steps[np.argmax(crossed)])

def bootstrap_ci(steps: np.ndarray, metric: np.ndarray, detector: Callable, n_resamples: int = 100, ci_level: float = 95.0, **kwargs) -> Tuple[Optional[float], Tuple[float, float]]:
    """
    Computes a bootstrap confidence interval for a detector estimator over resampled evaluation grids.

    Args:
        steps: Array of evaluation steps.
        metric: Array of metric values.
        detector: The detector function to estimate the grokking step.
        n_resamples: Number of bootstrap iterations.
        ci_level: Confidence interval level (e.g., 95.0 for 95% CI).

    Returns:
        A tuple of (point_estimate, (lower_bound, upper_bound)).
        If the point estimate fails, returns (None, (NaN, NaN)).
    """
    point_est = detector(steps, metric, **kwargs)
    if point_est is None:
        return None, (np.nan, np.nan)

    estimates = []
    n = len(steps)
    for _ in range(n_resamples):
        # Sample indices with replacement
        indices = np.random.choice(n, size=n, replace=True)
        # Sort indices to maintain timeline order for detectors that expect monotonic steps
        indices.sort()

        # Jitter duplicate steps slightly to avoid scipy curve_fit issues with non-unique x
        resampled_steps = steps[indices].astype(float)
        # Small jitter to steps so they are strictly increasing
        resampled_steps += np.random.normal(0, 1e-5, size=n)
        resampled_steps = np.sort(resampled_steps)

        resampled_metric = metric[indices]

        est = detector(resampled_steps, resampled_metric, **kwargs)
        if est is not None:
            estimates.append(est)

    if not estimates:
        return point_est, (np.nan, np.nan)

    alpha = (100.0 - ci_level) / 2.0
    lower = np.percentile(estimates, alpha)
    upper = np.percentile(estimates, 100.0 - alpha)

    return point_est, (float(lower), float(upper))
