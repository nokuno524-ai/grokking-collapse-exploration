import numpy as np
from typing import List, Dict, Tuple, Optional

def detect_grokking_transition(test_accuracy: List[float], steps: List[int], acc_threshold: float = 0.9, window_size: int = 5) -> Optional[int]:
    """
    Detect the grokking transition point where test accuracy rapidly increases.
    Identified as the first step where accuracy crosses the threshold AND has a positive derivative.

    Args:
        test_accuracy: List of test accuracy values over time.
        steps: Corresponding training steps.
        acc_threshold: The accuracy threshold defining successful learning.
        window_size: Size of window for smoothing derivative calculation.

    Returns:
        The training step where the grokking transition occurs, or None if not found.
    """
    if len(test_accuracy) < window_size:
        return None

    acc_array = np.array(test_accuracy)

    # Calculate smoothed derivative using central difference
    derivatives = np.zeros_like(acc_array)
    for i in range(1, len(acc_array) - 1):
        derivatives[i] = (acc_array[i+1] - acc_array[i-1]) / 2.0

    # Find points where accuracy crosses threshold
    above_thresh = acc_array >= acc_threshold

    # Check if threshold is ever crossed
    if not np.any(above_thresh):
        return None

    # Find the first index where accuracy is >= threshold and derivative is positive
    for i in range(1, len(acc_array)):
        if above_thresh[i] and not above_thresh[i-1] and derivatives[i] > 0:
            # Found a candidate transition. Verify it stays above threshold for a bit
            # (Grokking usually defined as staying above threshold for ~50 steps, but here we just check immediate stabilization)
            future_window = min(len(acc_array) - i, 5)
            if np.mean(acc_array[i:i+future_window]) >= acc_threshold - 0.05:
                return steps[i]

    # Fallback: just return the first step it crosses the threshold
    first_cross = np.argmax(above_thresh)
    if above_thresh[first_cross]:
         return steps[first_cross]

    return None

def detect_collapse_onset(weight_norms: List[float], steps: List[int], explosion_factor: float = 5.0, window_size: int = 10) -> Optional[int]:
    """
    Detect the onset of model collapse by monitoring weight norm explosion.
    Collapse is identified when weight norms increase by `explosion_factor` relative to moving average.

    Args:
        weight_norms: List of total weight norms over time.
        steps: Corresponding training steps.
        explosion_factor: Threshold factor for relative increase indicating collapse.
        window_size: Window size for moving average baseline.

    Returns:
        The training step where collapse begins, or None if not detected.
    """
    if len(weight_norms) < window_size + 1:
        return None

    norms = np.array(weight_norms)

    # Calculate rolling mean
    rolling_mean = np.zeros_like(norms)
    for i in range(len(norms)):
        start_idx = max(0, i - window_size)
        rolling_mean[i] = np.mean(norms[start_idx:i+1])

    # Detect sudden spikes
    for i in range(window_size, len(norms)):
        # Compare current norm to the rolling average from previous steps
        baseline = rolling_mean[i-1]
        if baseline > 0 and norms[i] / baseline > explosion_factor:
            return steps[i]

    return None

def identify_intervention_periods(metric_history: List[float], steps: List[int], threshold: float, condition: str = 'above') -> List[Tuple[int, int]]:
    """
    Identify continuous periods where a metric meets a certain condition (e.g., gradient noise is high).
    Useful for finding windows to apply interventions like regularization or learning rate changes.

    Args:
        metric_history: List of metric values (e.g., gradient noise, weight velocity).
        steps: Corresponding training steps.
        threshold: The threshold value.
        condition: 'above' or 'below' the threshold.

    Returns:
        List of tuples (start_step, end_step) representing continuous periods meeting the condition.
    """
    periods = []
    in_period = False
    start_step = None

    for i, val in enumerate(metric_history):
        meets_condition = (val >= threshold) if condition == 'above' else (val <= threshold)

        if meets_condition and not in_period:
            in_period = True
            start_step = steps[i]
        elif not meets_condition and in_period:
            in_period = False
            periods.append((start_step, steps[i-1]))

    # Handle period extending to the end
    if in_period:
        periods.append((start_step, steps[-1]))

    return periods
