import numpy as np

def detect_grokking_transition(test_accs: np.ndarray, threshold: float = 0.90) -> int:
    """
    Detect the exact grokking step by calculating the discrete derivative of the
    test accuracy curve.

    Returns:
        The step index where grokking occurs, or -1 if the maximum accuracy
        remains below the threshold.
    """
    if len(test_accs) == 0:
        return -1

    if np.max(test_accs) < threshold:
        return -1

    # Find points where accuracy is above threshold
    above_thresh = np.where(test_accs >= threshold)[0]

    if len(above_thresh) == 0:
        return -1

    # Find the earliest point
    grokking_idx = above_thresh[0]

    # Optional: ensure we look for the sharpest jump before this point
    # diffs = np.diff(test_accs[:grokking_idx+1])
    # if len(diffs) > 0:
    #     sharpest_jump = np.argmax(diffs)
    #     return int(sharpest_jump + 1)

    return int(grokking_idx)


def detect_collapse_onset(weight_norms: np.ndarray) -> int:
    """
    Detect the collapse onset (weight norm acceleration).
    Finds the index where the negative slope of weight norm accelerates the most.

    Returns:
        The step index of the onset, or -1 if array is too small.
    """
    if len(weight_norms) < 3:
        return -1

    # Calculate discrete derivative
    diffs = np.diff(weight_norms)

    # Calculate the second discrete derivative
    # (Acceleration of weight norm decrease)
    diff2 = np.diff(diffs)

    # The most negative second derivative is where the downward slope
    # becomes steepest most suddenly (acceleration of collapse)
    min_accel_idx = np.argmin(diff2)

    # The index returned is +1 because the first diff array is off by 1,
    # and second is off by 2.
    return int(min_accel_idx + 1)
