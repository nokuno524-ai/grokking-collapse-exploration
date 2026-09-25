import math
import numpy as np
from typing import List, Tuple, Optional, Callable
from scipy import optimize

def threshold_crossing_step(steps: List[int], accs: List[float], threshold: float = 0.95, sustained_steps: int = 1) -> Optional[int]:
    """
    Finds the first step where the test accuracy crosses and remains above the threshold.

    Args:
        steps: List of integer training steps.
        accs: List of float test accuracies.
        threshold: The accuracy threshold to cross.
        sustained_steps: The number of consecutive evaluation steps the accuracy must remain >= threshold.

    Returns:
        The step at which the threshold was initially crossed and sustained, or None.
    """
    if len(steps) != len(accs):
        raise ValueError("steps and accs must have the same length")
    if len(steps) == 0:
        return None

    consecutive = 0
    first_cross = None

    for step, acc in zip(steps, accs):
        if acc is None or math.isnan(acc):
            acc = 0.0

        if acc >= threshold - 1e-6:
            if consecutive == 0:
                first_cross = step
            consecutive += 1
            if consecutive >= sustained_steps:
                return first_cross
        else:
            consecutive = 0
            first_cross = None

    return None

def smooth_curve(y: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Applies a simple moving average filter.

    Args:
        y (np.ndarray): The input 1D array to smooth.
        window_size (int): The size of the moving average window.

    Returns:
        np.ndarray: The smoothed array, with length len(y) - window_size + 1.

    """
    if len(y) < window_size:
        return y
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

def max_2nd_derivative_detector(steps: List[int], accs: List[float], window_size: int = 5) -> Optional[int]:
    """
    Locates the start of the cliff via the maximum acceleration (2nd derivative)
    of the smoothed test accuracy.

    Args:
        steps: List of training steps.
        accs: List of test accuracies.
        window_size: Smoothing window size.

    Returns:
        The step number corresponding to the maximum 2nd derivative, or None.
    """
    if len(steps) < max(3, window_size):
        return None

    y = np.array(accs, dtype=float)
    y = np.where(np.isnan(y), 0.0, y)

    # Smooth
    y_smooth = smooth_curve(y, window_size)

    if len(y_smooth) < 3:
        return None

    # Calculate 2nd derivative using central differences (ignoring non-uniform step spacing for simplicity)
    # y_smooth length is len(y) - window_size + 1
    # 1st deriv length is len(y_smooth) - 1
    # 2nd deriv length is len(y_smooth) - 2
    d2y = np.diff(y_smooth, n=2)

    if len(d2y) == 0 or np.all(d2y == 0.0):
        return None

    # Find max 2nd derivative index
    max_idx = np.argmax(d2y)

    # Map back to original steps array
    # The convolution 'valid' shifts by (window_size - 1)
    # The 2nd derivative shifts by 2 (first element corresponds to original index 1 in the smoothed array)
    # For a causal-like interpretation where the change happens AT the step,
    # mapping to the center or start of the diff window is reasonable.
    # index in d2y -> index in y_smooth = max_idx + 1
    # index in y_smooth -> index in original = max_idx + 1 + (window_size - 1) // 2

    shift = (window_size - 1) // 2
    orig_idx = max_idx + 1 + shift

    # Ensure it's within bounds
    orig_idx = max(0, min(orig_idx, len(steps) - 1))
    return steps[orig_idx]

def _sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def changepoint_detector(steps: List[int], accs: List[float]) -> Optional[int]:
    """
    Fits a sigmoid curve to the test accuracy to find the midpoint of the transition.

    Args:
        steps: List of training steps.
        accs: List of test accuracies.

    Returns:
        The step closest to the sigmoid midpoint (x0), or None if fitting fails.
    """
    if len(steps) < 10:
        return None

    x = np.array(steps, dtype=float)
    y = np.array(accs, dtype=float)
    y = np.where(np.isnan(y), 0.0, y)

    # Normalize x to help with fitting
    x_min, x_max = x.min(), x.max()
    if x_min == x_max or np.var(y) < 1e-6:
        return None
    x_norm = (x - x_min) / (x_max - x_min)

    # Initial guesses: L=max(y), x0=mean(x_norm), k=10 (steepness), b=min(y)
    p0 = [max(y), np.mean(x_norm), 10.0, min(y)]
    bounds = (
        [0.0, 0.0, 0.0, 0.0],
        [1.1, 1.0, 1000.0, 1.0]
    )

    try:
        popt, _ = optimize.curve_fit(_sigmoid, x_norm, y, p0=p0, bounds=bounds, maxfev=2000)
        x0_norm = popt[1]
        x0 = x0_norm * (x_max - x_min) + x_min

        # Find closest step
        idx = (np.abs(x - x0)).argmin()
        return int(steps[idx])
    except (RuntimeError, optimize.OptimizeWarning, ValueError):
        return None

def compute_uncertainty_band(steps: List[int], accs: List[float], method: str = 'max_2nd_derivative', n_bootstraps: int = 0) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Computes the cliff step and an uncertainty band by varying smoothing parameters or bootstrapping.

    Args:
        steps: List of training steps.
        accs: List of test accuracies.
        method: The cliff detection method to use.
        n_bootstraps: If > 0, performs bootstrapping over accuracy measurements by adding normal noise proportional to variance to simulate seed variation.
    """
    if n_bootstraps > 0:
        detected = []
        accs_np = np.array(accs)
        std_dev = np.std(accs_np) if len(accs_np) > 1 else 0.05
        for _ in range(n_bootstraps):
            noise = np.random.normal(0, std_dev * 0.1, size=len(accs_np))
            noisy_accs = accs_np + noise
            noisy_accs = np.clip(noisy_accs, 0.0, 1.0)
            if method == 'max_2nd_derivative':
                s = max_2nd_derivative_detector(steps, noisy_accs.tolist())
            elif method == 'changepoint':
                s = changepoint_detector(steps, noisy_accs.tolist())
            else:
                s = threshold_crossing_step(steps, noisy_accs.tolist())
            if s is not None:
                detected.append(s)
        if detected:
            detected.sort()
            return detected[len(detected)//2], detected[int(0.05*len(detected))], detected[int(0.95*len(detected))]
        return None, None, None

    """
    Computes the cliff step and an uncertainty band by varying smoothing parameters.

    Args:
        steps: List of training steps.
        accs: List of test accuracies.
        method: The cliff detection method to use.

    Returns:
        A tuple of (median_step, lower_bound_step, upper_bound_step)
    """
    if method == 'max_2nd_derivative':
        windows = [3, 5, 7, 9, 11]
        detected = []
        for w in windows:
            s = max_2nd_derivative_detector(steps, accs, window_size=w)
            if s is not None:
                detected.append(s)
    elif method == 'changepoint':
        # for changepoint, we can subsample the data
        detected = []
        s = changepoint_detector(steps, accs)
        if s is not None:
            detected.append(s)

        # Try a few subsamples (every 2nd point, shifted)
        if len(steps) >= 20:
            s1 = changepoint_detector(steps[::2], accs[::2])
            if s1 is not None: detected.append(s1)
            s2 = changepoint_detector(steps[1::2], accs[1::2])
            if s2 is not None: detected.append(s2)
    else:
        # Default fallback to threshold crossing varying thresholds
        detected = []
        for t in [0.90, 0.95, 0.99]:
            s = threshold_crossing_step(steps, accs, threshold=t)
            if s is not None:
                detected.append(s)

    if not detected:
        return None, None, None

    detected.sort()
    median_val = detected[len(detected)//2]
    return median_val, detected[0], detected[-1]
