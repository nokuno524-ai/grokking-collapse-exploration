"""
Grokking cliff detectors with smoothing windows and multiple criteria.
"""

import numpy as np
from typing import Optional, List, Tuple

def _smooth_curve(curve: np.ndarray, window: int) -> np.ndarray:
    """Smooth a curve using a moving average window."""
    if window <= 1 or len(curve) < window:
        return curve
    # Use valid convolution mode to not introduce edge artifacts
    kernel = np.ones(window) / window
    smoothed = np.convolve(curve, kernel, mode='valid')
    # Pad beginning with first value to maintain length alignment
    pad_len = len(curve) - len(smoothed)
    if pad_len > 0:
        smoothed = np.concatenate((np.full(pad_len, smoothed[0]), smoothed))
    return smoothed

def fixed_threshold_detector(steps: List[int], values: List[float], threshold: float, window: int = 1) -> Optional[int]:
    """
    Detect the first step where the (smoothed) value crosses a threshold.
    """
    if not values:
        return None
    smoothed = _smooth_curve(np.array(values), window)
    for i, val in enumerate(smoothed):
        if val >= threshold:
            return steps[i]
    return None

def max_2nd_derivative_detector(steps: List[int], values: List[float], window: int = 1) -> Optional[int]:
    """
    Detect grokking step as the point of maximum positive acceleration (2nd derivative)
    of the smoothed curve.
    """
    if len(values) < 3:
        return None
    smoothed = _smooth_curve(np.array(values), window)
    # Compute 1st and 2nd derivatives. Since steps might not be strictly uniform,
    # we approximate with finite differences on the indices.
    # We really just want the index where the value spikes up fastest.
    d1 = np.diff(smoothed)
    d2 = np.diff(d1)

    if len(d2) == 0:
        return None

    # Find max 2nd derivative (acceleration)
    # The length of d2 is len(values) - 2. We shift by 1 to roughly center it.
    idx = np.argmax(d2) + 1

    # We want to ensure it's actually an upward spike, not just noise.
    # Require some minimum overall change.
    if np.max(smoothed) - np.min(smoothed) < 0.1:
        return None

    return steps[idx]

def plateau_detector(steps: List[int], values: List[float], window: int = 1, patience: int = 5, plateau_threshold: float = 0.9, noise_tol: float = 0.05) -> Optional[int]:
    """
    Detect when the curve reaches a high level (plateau_threshold) and stays within
    noise_tol for 'patience' steps.
    """
    if not values or len(values) < patience:
        return None
    smoothed = _smooth_curve(np.array(values), window)

    for i in range(len(smoothed) - patience + 1):
        window_vals = smoothed[i:i+patience]
        if np.all(window_vals >= plateau_threshold - noise_tol):
            return steps[i]
    return None
