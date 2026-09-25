"""
Robust grokking cliff detection module.
"""

import numpy as np

def smooth_curve(steps, values, window=5, method='moving_average'):
    """
    Smooth a noisy training curve.

    Args:
        steps (list or np.ndarray): x-axis steps.
        values (list or np.ndarray): y-axis metric values.
        window (int): Size of the smoothing window.
        method (str): Smoothing method ('moving_average').

    Returns:
        np.ndarray: Smoothed values.
    """
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values

    if method == 'moving_average':
        kernel = np.ones(window) / window
        # Use 'same' to keep length consistent, pad at ends
        smoothed = np.convolve(values, kernel, mode='same')
        # Fix boundary issues for 'same' mode
        pad = window // 2
        for i in range(pad):
            if i > 0:
                smoothed[i] = np.mean(values[:i+pad+1])
                smoothed[-i-1] = np.mean(values[-i-pad-1:])
            else:
                smoothed[0] = values[0]
                smoothed[-1] = values[-1]
        return smoothed
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

def threshold_detector(steps, values, threshold=0.95):
    """
    Detect cliff based on crossing a threshold.
    """
    steps = np.array(steps)
    values = np.array(values)

    crossing_idx = np.where(values >= threshold)[0]
    if len(crossing_idx) == 0:
        return None

    return int(steps[crossing_idx[0]])

def changepoint_detector(steps, values):
    """
    Detect cliff based on maximum first derivative.
    """
    steps = np.array(steps)
    values = np.array(values)

    if len(values) < 2:
        return None

    deriv = np.diff(values) / np.diff(steps)
    max_idx = np.argmax(deriv)

    return int(steps[max_idx + 1])

def detect_cliff(steps, values, method='threshold', threshold=0.95, window=5, smooth_method='moving_average'):
    """
    Detect a cliff in a noisy training curve.

    Args:
        steps (list or np.ndarray): x-axis steps.
        values (list or np.ndarray): y-axis metric values.
        method (str): Detection method ('threshold' or 'changepoint').
        threshold (float): Threshold for 'threshold' method.
        window (int): Size of smoothing window.
        smooth_method (str): Smoothing method.

    Returns:
        dict: Information about the detected cliff, or None if no cliff is found.
    """
    steps = np.array(steps)
    values = np.array(values)

    if len(steps) < 2:
        return None

    smoothed = smooth_curve(steps, values, window=window, method=smooth_method)

    cliff_step = None
    if method == 'threshold':
        cliff_step = threshold_detector(steps, smoothed, threshold=threshold)
    elif method == 'changepoint':
        cliff_step = changepoint_detector(steps, smoothed)
    else:
        raise ValueError(f"Unknown detection method: {method}")

    if cliff_step is None:
        return None

    idx = np.where(steps == cliff_step)[0][0]

    # Calculate pre and post cliff levels (mean of window around cliff)
    pre_window = max(0, idx - window)
    post_window = min(len(values), idx + window + 1)

    pre_level = float(np.mean(values[pre_window:idx])) if idx > pre_window else float(values[idx])
    post_level = float(np.mean(values[idx+1:post_window])) if post_window > idx + 1 else float(values[idx])

    # Sharpness: derivative at the cliff
    if idx > 0 and idx < len(values) - 1:
        sharpness = float((smoothed[idx+1] - smoothed[idx-1]) / (steps[idx+1] - steps[idx-1]))
    else:
        sharpness = 0.0

    # Confidence: how clear the cliff is (heuristic: difference between pre and post levels relative to variance)
    level_diff = post_level - pre_level
    var_pre = np.var(values[pre_window:idx]) if idx > pre_window + 1 else 1e-6
    var_post = np.var(values[idx+1:post_window]) if post_window > idx + 2 else 1e-6
    confidence = float(level_diff / np.sqrt(var_pre + var_post + 1e-6))

    return {
        'cliff_step': int(cliff_step),
        'pre_cliff_level': pre_level,
        'post_cliff_level': post_level,
        'sharpness': sharpness,
        'confidence': confidence
    }
