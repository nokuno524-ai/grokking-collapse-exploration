"""
Robust grok-step changepoint detection.
Uses automated sigmoid curve fitting and derivative detectors.
"""

import numpy as np
from scipy.optimize import curve_fit

def _sigmoid(x, L, x0, k, b):
    # Using np.clip to prevent overflow errors in np.exp
    z = np.clip(-k * (x - x0), -500, 500)
    return L / (1 + np.exp(z)) + b

def fit_sigmoid_transition(steps: np.ndarray, metric: np.ndarray) -> float:
    """Fits a sigmoid to the metric to find the transition step (x0)."""
    if len(steps) < 4:
        return float('nan')
    try:
        p0 = [max(metric), np.median(steps), 1, min(metric)]
        popt, _ = curve_fit(_sigmoid, steps, metric, p0=p0, maxfev=10000)
        return float(popt[1])
    except RuntimeError:
        return float('nan')

def max_2nd_derivative_detector(steps: np.ndarray, metric: np.ndarray, window: int = 5) -> float:
    """Finds the maximum 2nd derivative of the smoothed metric."""
    if len(steps) < window * 2:
        return float('nan')
    smoothed = np.convolve(metric, np.ones(window)/window, mode='valid')
    smoothed_steps = steps[window//2 : -window//2 + 1] if window % 2 != 0 else steps[window//2 - 1 : -window//2]
    # Ensure dimensions match
    min_len = min(len(smoothed), len(smoothed_steps))
    smoothed = smoothed[:min_len]
    smoothed_steps = smoothed_steps[:min_len]

    first_deriv = np.gradient(smoothed, smoothed_steps)
    second_deriv = np.gradient(first_deriv, smoothed_steps)
    idx = np.argmax(np.abs(second_deriv))
    return float(smoothed_steps[idx])

def plateau_detector(steps: np.ndarray, metric: np.ndarray, threshold: float = 0.95, window: int = 5) -> float:
    """Finds the first step where the smoothed metric is >= threshold."""
    if len(steps) < window:
        return float('nan')
    smoothed = np.convolve(metric, np.ones(window)/window, mode='valid')
    smoothed_steps = steps[window//2 : window//2 + len(smoothed)]

    for s, m in zip(smoothed_steps, smoothed):
        if m >= threshold - 1e-6:
            return float(s)
    return float('nan')
