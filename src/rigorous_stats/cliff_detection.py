import numpy as np
from typing import Tuple, Optional, List

def binary_segmentation_changepoint(y: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """
    Find a single changepoint that minimizes the sum of squared errors
    of piecewise constant fits.

    This implements a basic cost-based binary segmentation algorithm without
    external dependencies beyond numpy. It models the sequence as two segments,
    each with a constant mean, and finds the split point that minimizes the
    total variance.

    Args:
        y: 1D numpy array of values (e.g., test accuracies).

    Returns:
        idx: Index of the changepoint (the first element of the second segment).
             Returns 0 if no valid changepoint can be found (e.g. array too short).
        best_cost: The minimal cost found at the changepoint.
        costs: Array of shape (len(y),) containing the cost for splitting at each index.
               Indices 0 and len(y) have cost infinity.
    """
    n = len(y)
    costs = np.full(n, np.inf)

    # Handle edge cases (empty, single item, NaNs)
    if n < 2 or np.any(np.isnan(y)):
        return 0, 0.0, costs

    # We want to find index c in [1, n-1] to split into y[:c] and y[c:]
    # Calculate prefix sums and prefix sum of squares for fast variance computation
    s1 = np.cumsum(y)
    s2 = np.cumsum(y**2)

    best_cost = np.inf
    best_idx = 0

    for c in range(1, n):
        n1 = c
        n2 = n - c

        sum1 = s1[c-1]
        sum2 = s1[-1] - s1[c-1]

        sqsum1 = s2[c-1]
        sqsum2 = s2[-1] - s2[c-1]

        # Variance of segment = E[X^2] - (E[X])^2 = (Sum_sq / n) - (Sum / n)^2
        # Sum of squared errors for segment = Sum_sq - (Sum^2) / n
        var1 = sqsum1 - (sum1**2) / n1
        var2 = sqsum2 - (sum2**2) / n2

        cost = var1 + var2
        costs[c] = cost

        if cost < best_cost:
            best_cost = cost
            best_idx = c

    return best_idx, best_cost, costs

def find_cliff_location(steps: np.ndarray, accuracies: np.ndarray, margin_factor: float = 1.05) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    Finds the grokking cliff location step with an uncertainty interval.

    Uses binary segmentation to find the changepoint. The uncertainty interval
    is defined by the contiguous set of steps around the optimal changepoint
    whose cost is within `margin_factor` times the minimum cost.

    Args:
        steps: 1D numpy array of step numbers.
        accuracies: 1D numpy array of corresponding test accuracies.
        margin_factor: Defines the threshold for the uncertainty interval.
                       Costs <= min_cost * margin_factor are included in the interval.

    Returns:
        cliff_step: The step where the cliff begins, or None if no valid cliff found.
        ci: A tuple (lower_step, upper_step) representing the uncertainty interval, or None.
            If a cliff is found, this interval will contain at least the cliff_step.
    """
    # Input validation
    if len(accuracies) != len(steps):
        raise ValueError("steps and accuracies must have the same length")

    n = len(accuracies)
    if n < 3 or np.any(np.isnan(accuracies)):
        return None, None

    idx, min_cost, costs = binary_segmentation_changepoint(accuracies)

    # If changepoint is at the very beginning or end, it's not a real cliff
    if idx == 0 or idx == n:
        return None, None

    # Check if this is a grokking cliff (sharp increase)
    pre_mean = np.mean(accuracies[:idx])
    post_mean = np.mean(accuracies[idx:])

    # Require some minimum jump to be considered a cliff (e.g. going from random chance to memorization)
    if post_mean - pre_mean < 0.2:
        return None, None

    # Define uncertainty interval based on the cost landscape
    # We find the widest contiguous region around `idx` where cost <= threshold
    threshold_cost = min_cost * margin_factor if min_cost > 0 else 0.0
    if min_cost == 0:
        # Exact match piecewise constant, all 0 cost regions are valid
        threshold_cost = 1e-10

    left_idx = idx
    while left_idx > 1 and costs[left_idx - 1] <= threshold_cost:
        left_idx -= 1

    right_idx = idx
    while right_idx < n - 1 and costs[right_idx + 1] <= threshold_cost:
        right_idx += 1

    cliff_step = float(steps[idx])
    ci = (float(steps[left_idx]), float(steps[right_idx]))

    return cliff_step, ci
