"""
Methods for detecting grokking cliffs and changepoints in training trajectories.
"""

from typing import Optional, Tuple, List
import numpy as np
import scipy.optimize

def _smooth(data: np.ndarray, window: int) -> np.ndarray:
    """Apply a simple moving average filter."""
    if window <= 1 or len(data) < window:
        return data
    kernel = np.ones(window) / window
    # Valid mode means the output is smaller, pad it or use a different mode
    # Let's use mode='same' equivalent by padding
    pad_width = window // 2
    padded = np.pad(data, (pad_width, window - 1 - pad_width), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def plateau_detector(
    steps: np.ndarray,
    accs: np.ndarray,
    threshold: float = 0.95,
    window: int = 1
) -> Optional[int]:
    """
    Detect the first step where accuracy stays above threshold for the rest of training.
    """
    if len(steps) == 0:
        return None

    smoothed_accs = _smooth(accs, window)

    # Find indices where accuracy is below threshold
    below_thresh = np.where(smoothed_accs < threshold)[0]

    if len(below_thresh) == 0:
        return int(steps[0])

    # The first index where it stays above threshold is one after the last time it was below
    last_below_idx = below_thresh[-1]

    if last_below_idx == len(steps) - 1:
        return None # Never reliably reaches threshold

    return int(steps[last_below_idx + 1])


def max_2nd_derivative_detector(
    steps: np.ndarray,
    accs: np.ndarray,
    window: int = 5
) -> Optional[int]:
    """
    Detect the inflection point by finding the maximum second derivative of the smoothed curve.
    Requires at least 3 points. Returns the step where the transition is fastest.
    """
    if len(steps) < 3:
        return None

    smoothed = _smooth(accs, window)

    # We can assume steps are uniformly spaced for the derivative calculation
    # or just compute simple differences
    dy = np.diff(smoothed)
    dx = np.diff(steps)
    first_deriv = dy / dx

    # Center points for the first derivative
    mid_steps = (steps[:-1] + steps[1:]) / 2

    d2y = np.diff(first_deriv)
    d2x = np.diff(mid_steps)
    second_deriv = d2y / d2x

    if len(second_deriv) == 0:
        return None

    # The cliff start is the point of maximum acceleration (max 2nd derivative).
    max_idx = np.argmax(second_deriv)
    # second_deriv has length len(steps) - 2.
    # The index maps to mid_steps[:-1], which roughly maps to steps[max_idx + 1]
    return int(steps[max_idx + 1])


def fit_sigmoid_transition(
    steps: np.ndarray,
    accs: np.ndarray
) -> Tuple[Optional[int], dict]:
    """
    Fit a generalized sigmoid curve to the accuracy data and extract the midpoint parameter.
    Sigmoid: L / (1 + exp(-k * (x - x0))) + b
    """
    if len(steps) < 4:
        return None, {}

    def sigmoid(x, L, k, x0, b):
        # Clip x to avoid overflow in exp
        x_shifted = np.clip(k * (x - x0), -100, 100)
        return L / (1 + np.exp(-x_shifted)) + b

    # Initial guesses
    L_guess = max(np.max(accs) - np.min(accs), 0.1) # Amplitude
    b_guess = np.min(accs) # Baseline

    # Guess midpoint: where it crosses baseline + amplitude/2
    mid_val = b_guess + L_guess / 2
    idx = np.abs(accs - mid_val).argmin()
    x0_guess = steps[idx]

    k_guess = 1.0 / (steps[-1] - steps[0] + 1e-5) # Slope guess

    bounds = (
        [0, 0, steps[0], 0], # L, k, x0, b
        [1.5, np.inf, steps[-1], 1.0]
    )

    try:
        popt, _ = scipy.optimize.curve_fit(
            sigmoid,
            steps,
            accs,
            p0=[L_guess, k_guess, x0_guess, b_guess],
            bounds=bounds,
            maxfev=10000
        )
        L, k, x0, b = popt

        # If the amplitude is very small, there's no real transition
        if L < 0.1:
            return None, {"L": L, "k": k, "x0": x0, "b": b, "reason": "Amplitude too small"}

        return int(x0), {"L": L, "k": k, "x0": x0, "b": b}

    except (RuntimeError, scipy.optimize.OptimizeWarning):
        return None, {"reason": "Curve fit failed"}
