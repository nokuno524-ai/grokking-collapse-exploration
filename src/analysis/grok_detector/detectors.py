import numpy as np
from typing import List, Tuple, Optional, Dict, Any


def smooth_series(series: List[float], window_size: int) -> np.ndarray:
    """
    Smooth a 1D time series using a simple moving average (boxcar filter).

    Args:
        series: A list or 1D numpy array of floats.
        window_size: The number of adjacent points to average. Must be >= 1.
                     If window_size is 1, returns the original series.

    Returns:
        A smoothed numpy array of the same length as the input series.
        Edges are padded by replicating the first/last elements to avoid boundary effects.
    """
    arr = np.array(series, dtype=float)
    if window_size <= 1:
        return arr

    if len(arr) == 0:
        return arr

    if window_size > len(arr):
        window_size = len(arr)

    # Pad the array to handle edges
    pad_width = window_size // 2
    padded = np.pad(arr, (pad_width, window_size - 1 - pad_width), mode='edge')

    # Compute moving average using convolution
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')

    return smoothed


def max_2nd_derivative_detector(
    steps: List[int],
    metric_values: List[float],
    window_size: int = 5,
    threshold: float = 0.95,
    min_acc_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Detects the grokking cliff as the point of maximum acceleration (max 2nd derivative)
    in the smoothed test accuracy trajectory.

    Criteria for "no grokking within budget":
    - The maximum smoothed metric value never reaches `threshold` (e.g., 0.95).
    - The maximum metric value never exceeds `min_acc_threshold` (e.g. 0.5) to avoid finding noise peaks.

    Args:
        steps: List of integer training steps.
        metric_values: List of metric values (e.g., test accuracy) at those steps.
        window_size: Size of the smoothing window to reduce noise before computing derivatives.
        threshold: The absolute threshold the metric must eventually cross to consider it grokked.
        min_acc_threshold: A lower bound threshold to filter out noise when no real grokking happens.

    Returns:
        A dictionary containing:
            - 'grokked': bool, True if grokking occurred.
            - 'cliff_step': int or None, the step at which the max 2nd derivative occurred.
            - 'peak_width': int or None, an estimate of uncertainty based on the full width at half max of the 2nd derivative peak.
    """
    if len(steps) < 3:
        return {'grokked': False, 'cliff_step': None, 'peak_width': None}

    steps_arr = np.array(steps)
    vals_arr = np.array(metric_values)

    # Smooth the values
    smoothed_vals = smooth_series(vals_arr, window_size)

    # Check for "no grokking" criteria
    if np.max(smoothed_vals) < threshold or np.max(smoothed_vals) < min_acc_threshold:
        return {'grokked': False, 'cliff_step': None, 'peak_width': None}

    # Calculate first derivative (velocity)
    # Using central differences for interior points, forward/backward for edges
    d1 = np.gradient(smoothed_vals, steps_arr)

    # Smooth first derivative before taking second derivative to reduce noise amplification
    d1_smoothed = smooth_series(d1, window_size)

    # Calculate second derivative (acceleration)
    d2 = np.gradient(d1_smoothed, steps_arr)

    # Find the maximum of the second derivative
    # Only consider points where the metric hasn't already saturated
    mask = smoothed_vals < threshold
    if not np.any(mask):
        return {'grokked': False, 'cliff_step': None, 'peak_width': None}

    masked_d2 = np.where(mask, d2, -np.inf)
    max_idx = np.argmax(masked_d2)
    cliff_step = int(steps_arr[max_idx])
    max_d2_val = d2[max_idx]

    # Estimate uncertainty (peak width at half maximum)
    half_max = max_d2_val / 2.0

    # Find points on the left and right that cross half_max
    left_idx = max_idx
    while left_idx > 0 and d2[left_idx] > half_max:
        left_idx -= 1

    right_idx = max_idx
    while right_idx < len(d2) - 1 and d2[right_idx] > half_max:
        right_idx += 1

    peak_width = int(steps_arr[right_idx] - steps_arr[left_idx])

    return {
        'grokked': True,
        'cliff_step': cliff_step,
        'peak_width': peak_width
    }
