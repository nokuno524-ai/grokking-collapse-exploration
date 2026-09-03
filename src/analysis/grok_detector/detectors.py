import numpy as np
from typing import Tuple, List, Optional

def piecewise_linear_detector(steps: np.ndarray, accuracies: np.ndarray) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """
    Fits two disjoint linear segments over candidate breakpoints and finds the breakpoint
    that minimizes the residual sum of squares (RSS).

    Returns:
        breakpoint_step: The step corresponding to the detected changepoint, or None.
        uncertainty_band: A tuple (min_step, max_step) of steps with RSS close to minimum, or None.
    """
    if len(steps) != len(accuracies):
        raise ValueError("steps and accuracies must have the same length")

    n = len(steps)
    if n < 4:
        return None, None

    # Handle NaNs
    valid = ~np.isnan(accuracies)
    if not np.all(valid):
        steps = steps[valid]
        accuracies = accuracies[valid]
        n = len(steps)
        if n < 4:
            return None, None

    rss_values = np.full(n, np.inf)

    # Need at least 2 points for a line in each segment
    for b in range(1, n - 2):
        x1, y1 = steps[:b+1], accuracies[:b+1]
        x2, y2 = steps[b+1:], accuracies[b+1:]

        # Fit line 1
        A1 = np.vstack([x1, np.ones(len(x1))]).T
        m1, c1 = np.linalg.lstsq(A1, y1, rcond=None)[0]
        rss1 = np.sum((y1 - (m1 * x1 + c1)) ** 2)

        # Fit line 2
        A2 = np.vstack([x2, np.ones(len(x2))]).T
        m2, c2 = np.linalg.lstsq(A2, y2, rcond=None)[0]
        rss2 = np.sum((y2 - (m2 * x2 + c2)) ** 2)

        rss_values[b] = rss1 + rss2

    best_b = np.argmin(rss_values)
    if rss_values[best_b] == np.inf:
        return None, None

    min_rss = rss_values[best_b]

    # Calculate uncertainty band (e.g., RSS within 5% of min_rss, or absolute small threshold)
    # If min_rss is very close to 0, add a small epsilon to avoid empty bands
    threshold = min_rss * 1.05 + 1e-9
    close_bs = np.where(rss_values <= threshold)[0]

    return int(steps[best_b]), (int(steps[close_bs[0]]), int(steps[close_bs[-1]]))


def cusum_detector(steps: np.ndarray, accuracies: np.ndarray) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """
    CUSUM-based algorithm to detect the shift in mean accuracy from chance level.
    For an upward shift, the changepoint is the minimum of the cumulative sum of residuals.

    Returns:
        breakpoint_step: The step corresponding to the detected changepoint, or None.
        uncertainty_band: A tuple (min_step, max_step) based on curvature, or None.
    """
    if len(steps) != len(accuracies):
        raise ValueError("steps and accuracies must have the same length")

    n = len(steps)
    if n < 2:
        return None, None

    valid = ~np.isnan(accuracies)
    if not np.all(valid):
        steps = steps[valid]
        accuracies = accuracies[valid]
        n = len(steps)
        if n < 2:
            return None, None

    mean_total = np.mean(accuracies)
    residuals = accuracies - mean_total
    cusum = np.cumsum(residuals)

    # For a step-up, the CUSUM curve typically goes down and then up.
    # The changepoint is the minimum.
    best_b = np.argmin(cusum)

    # Uncertainty band: where cusum is close to the minimum
    # Let's say within 5% of the range of cusum
    cusum_range = np.max(cusum) - np.min(cusum)
    if cusum_range < 1e-9:
        return int(steps[best_b]), (int(steps[best_b]), int(steps[best_b]))

    threshold = cusum[best_b] + 0.05 * cusum_range
    close_bs = np.where(cusum <= threshold)[0]

    return int(steps[best_b]), (int(steps[close_bs[0]]), int(steps[close_bs[-1]]))


def detect_grokking_step(steps: np.ndarray, accuracies: np.ndarray,
                        chance_level: float = 1/59,
                        grok_threshold: float = 0.95) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """
    Combined selection rule to detect the grokking step.
    Handles boundaries (flat chance-level curves, monotone, etc).
    """
    valid = ~np.isnan(accuracies)
    steps = steps[valid]
    accuracies = accuracies[valid]

    if len(steps) < 4:
        return None, None

    max_acc = np.max(accuracies)
    min_acc = np.min(accuracies)

    # Check 1: Flat chance-level curve (never groks)
    if max_acc < grok_threshold:
        # Never reaches grokking threshold
        return None, None

    # Check 2: Already grokked from the start (or extremely fast)
    if min_acc >= grok_threshold:
        return int(steps[0]), (int(steps[0]), int(steps[0]))

    # Get proposals from both detectors
    pw_step, pw_band = piecewise_linear_detector(steps, accuracies)
    cusum_step, cusum_band = cusum_detector(steps, accuracies)

    # CUSUM is generally more robust to single sharp jumps, but piecewise linear
    # handles continuous but distinct slopes well.
    # We will trust CUSUM primarily, but validate it.

    best_step = None
    best_band = None

    if cusum_step is not None:
        best_step = cusum_step
        best_band = cusum_band
    elif pw_step is not None:
        best_step = pw_step
        best_band = pw_band
    else:
        # Fallback to simple threshold crossing if detectors fail
        crossing_idx = np.where(accuracies >= grok_threshold)[0]
        if len(crossing_idx) > 0:
            best_step = int(steps[crossing_idx[0]])
            best_band = (best_step, best_step)
            return best_step, best_band
        return None, None

    # Verify that at the detected step, the accuracy is actually changing
    # If the detected step is too late (e.g., accuracy already 0.99 for many steps),
    # we should correct it to the first crossing.
    crossing_idx = np.where(accuracies >= grok_threshold)[0]
    first_crossing = int(steps[crossing_idx[0]])

    # If our detected changepoint is WAY after the first time it crossed the threshold,
    # then it might have detected a plateau. CUSUM can sometimes put the min right before
    # the jump, so it's usually before or close to first_crossing.
    if best_step > first_crossing:
        best_step = first_crossing
        best_band = (first_crossing, first_crossing)

    return best_step, best_band
