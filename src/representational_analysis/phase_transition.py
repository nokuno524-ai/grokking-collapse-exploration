import numpy as np
from typing import Tuple, List, Dict
import scipy.stats as stats
from scipy.optimize import curve_fit

def detect_change_point(metrics_series: np.ndarray, window_size: int = 5) -> int:
    """
    Detect the index of a significant change point in a metric time series
    (e.g., test accuracy jumping during grokking).
    Uses a rolling window derivative approach.

    Args:
        metrics_series: 1D numpy array of metric values over time
        window_size: Size of the smoothing window

    Returns:
        int: Index of the detected change point (returns -1 if none found)
    """
    if len(metrics_series) < window_size * 2:
        return -1

    # Smooth the series
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(metrics_series, kernel, mode='valid')

    # First derivative (rate of change)
    diff = np.diff(smoothed)

    # Second derivative (acceleration)
    diff2 = np.diff(diff)

    # Grokking is characterized by a sudden, large increase
    # We look for the maximum positive acceleration followed by high velocity

    # Find points where diff is significantly positive (e.g., > 10% change)
    threshold = 0.05 # 5% change as a baseline heuristic
    candidates = np.where(diff > threshold)[0]

    if len(candidates) == 0:
        return -1

    # The true change point is typically the start of the steep climb
    change_idx = candidates[0]

    # Adjust for convolution and diff offsets
    actual_idx = change_idx + window_size // 2
    return actual_idx

def compute_derivative_metrics(metrics_series: np.ndarray, time_steps: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute first and second derivatives of training metrics with respect to time/steps.

    Args:
        metrics_series: 1D array of metric values
        time_steps: 1D array of corresponding training steps

    Returns:
        Dict containing 'velocity' (d/dt) and 'acceleration' (d^2/dt^2) arrays
    """
    if len(metrics_series) != len(time_steps) or len(metrics_series) < 3:
        raise ValueError("Series must have matching lengths and at least 3 points")

    # Compute dt
    dt = np.diff(time_steps)
    dt[dt == 0] = 1.0 # Prevent division by zero

    # First derivative (velocity)
    dydt = np.diff(metrics_series) / dt

    # Second derivative (acceleration)
    # We compute this at the midpoints of the first derivative
    dt2 = (dt[1:] + dt[:-1]) / 2
    dt2[dt2 == 0] = 1.0
    d2ydt2 = np.diff(dydt) / dt2

    # Pad to match original length (approximate)
    velocity = np.pad(dydt, (1, 0), mode='edge')
    acceleration = np.pad(d2ydt2, (1, 1), mode='edge')

    return {
        'velocity': velocity,
        'acceleration': acceleration
    }

def piecewise_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Perform statistical test for phase transition using piecewise linear regression
    with a single breakpoint.

    Model: y = a1*x + b1 if x < c else a2*x + b2
    with continuity constraint: a1*c + b1 = a2*c + b2

    Args:
        x: Independent variable (e.g., training steps)
        y: Dependent variable (e.g., log test loss or test accuracy)

    Returns:
        breakpoint (c): The estimated x-value of the phase transition
        params: Dict of fitted parameters {a1, b1, a2, b2, r_squared}
    """

    # Exhaustive search for the breakpoint
    best_x0 = None
    best_r2 = -float('inf')
    best_params = {}

    n = len(x)
    if n < 4:
        return -1.0, {'a1': 0, 'b1': 0, 'a2': 0, 'b2': 0, 'r_squared': 0}

    for i in range(2, n - 2):
        x0 = x[i]

        # Split data
        x1, y1 = x[:i], y[:i]
        x2, y2 = x[i:], y[i:]

        # Fit lines
        try:
            a1, b1 = np.polyfit(x1, y1, 1)
            a2, b2 = np.polyfit(x2, y2, 1)

            # Compute R^2
            y_pred = np.concatenate([a1 * x1 + b1, a2 * x2 + b2])
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))

            if r_squared > best_r2:
                best_r2 = r_squared
                best_x0 = x0
                best_params = {
                    'a1': float(a1),
                    'b1': float(b1),
                    'a2': float(a2),
                    'b2': float(b2),
                    'r_squared': float(r_squared)
                }
        except:
            continue

    if best_x0 is None:
        return -1.0, {'a1': 0, 'b1': 0, 'a2': 0, 'b2': 0, 'r_squared': 0}

    return float(best_x0), best_params
