import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Optional, Tuple, Any

def sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Sigmoid function for fitting learning curves."""
    # Clip to avoid overflow in exp
    z = np.clip(-k * (x - x0), -500, 500)
    return L / (1.0 + np.exp(z)) + b

def fit_sigmoid_transition(
    steps: np.ndarray,
    metrics: np.ndarray,
    bounds: Optional[Tuple[List[float], List[float]]] = None
) -> Dict[str, Any]:
    """
    Fits a sigmoid curve to the given trajectory to detect phase transitions.
    Returns transition center, width, and uncertainty metrics.

    Args:
        steps: Array of training steps
        metrics: Array of metric values (e.g., test accuracy)
        bounds: Optional bounds for the fit parameters (L, k, x0, b)

    Returns:
        Dict containing transition center, width, grokking_step (start of phase transition), and fit parameters.
    """
    if len(steps) < 4:
        return {"success": False, "error": "Not enough points to fit sigmoid"}

    # Default bounds if none provided
    if bounds is None:
        max_step = np.max(steps)
        min_step = np.min(steps)
        bounds = (
            [0.0, 0.0, min_step, -1.0],  # lower bounds (L, k, x0, b)
            [2.0, 1.0, max_step * 2, 1.0]   # upper bounds (L, k, x0, b)
        )

    # Initial guess
    # L: max - min metric
    # k: steepness (guess 0.01)
    # x0: step where metric is closest to median
    # b: min metric
    L_guess = np.max(metrics) - np.min(metrics)
    b_guess = np.min(metrics)
    x0_guess = steps[np.argmin(np.abs(metrics - (L_guess / 2 + b_guess)))]

    # Clip guesses to bounds just to be safe
    L_guess = np.clip(L_guess, bounds[0][0] + 1e-5, bounds[1][0] - 1e-5)
    b_guess = np.clip(b_guess, bounds[0][3] + 1e-5, bounds[1][3] - 1e-5)
    x0_guess = np.clip(x0_guess, bounds[0][2] + 1e-5, bounds[1][2] - 1e-5)

    p0 = [L_guess, 0.01, x0_guess, b_guess]

    try:
        popt, pcov = curve_fit(
            sigmoid,
            steps,
            metrics,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )

        L, k, x0, b = popt

        # Calculate standard errors from covariance matrix
        perr = np.sqrt(np.diag(pcov))

        # Transition width is often defined inversely proportional to k
        width = 4 / k if k > 1e-10 else float('inf')

        return {
            "success": True,
            "center": x0,
            "center_err": perr[2],
            "width": width,
            "L": L,
            "k": k,
            "b": b,
            "cov": pcov.tolist(),
            "perr": perr.tolist()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def max_2nd_derivative_detector(steps: np.ndarray, metrics: np.ndarray, window: int = 5) -> Dict[str, Any]:
    """
    Baseline detector: finds transition based on max derivative threshold.
    """
    if len(steps) < window:
         return {"success": False, "error": "Not enough points"}

    # Smooth the metrics slightly
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(metrics, kernel, mode='valid')
        valid_steps = steps[window-1:]
    else:
        smoothed = metrics
        valid_steps = steps

    if len(valid_steps) < 3:
        return {"success": False, "error": "Not enough points after smoothing"}

    # Central difference for derivative
    # df/dx = (f(x+h) - f(x-h)) / (2h)
    diff = np.zeros(len(smoothed))

    # Internal points
    diff[1:-1] = (smoothed[2:] - smoothed[:-2]) / (valid_steps[2:] - valid_steps[:-2])
    # Edges
    diff[0] = (smoothed[1] - smoothed[0]) / (valid_steps[1] - valid_steps[0])
    diff[-1] = (smoothed[-1] - smoothed[-2]) / (valid_steps[-1] - valid_steps[-2])

    max_idx = np.argmax(diff)

    return {
        "success": True,
        "center": valid_steps[max_idx],
        "max_derivative": diff[max_idx],
        "all_derivatives": diff.tolist(),
        "valid_steps": valid_steps.tolist()
    }
