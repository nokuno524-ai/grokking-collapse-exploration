import numpy as np
from typing import Tuple, Callable, Optional

def piecewise_linear_detector(steps: np.ndarray, values: np.ndarray) -> Optional[float]:
    """
    Detect a single changepoint by finding the breakpoint that minimizes
    the sum of squared errors of a two-segment piecewise linear fit.
    """
    if len(steps) < 4:
        return None

    best_sse = float('inf')
    best_step = None

    # Require at least 2 points in each segment
    for i in range(2, len(steps) - 2):
        # Left segment
        x1, y1 = steps[:i], values[:i]
        # Right segment
        x2, y2 = steps[i:], values[i:]

        # Fit left
        A1 = np.vstack([x1, np.ones(len(x1))]).T
        m1, c1 = np.linalg.lstsq(A1, y1, rcond=None)[0]
        sse1 = np.sum((y1 - (m1 * x1 + c1)) ** 2)

        # Fit right
        A2 = np.vstack([x2, np.ones(len(x2))]).T
        m2, c2 = np.linalg.lstsq(A2, y2, rcond=None)[0]
        sse2 = np.sum((y2 - (m2 * x2 + c2)) ** 2)

        total_sse = sse1 + sse2
        if total_sse < best_sse:
            best_sse = total_sse
            best_step = float(steps[i])

    return best_step

def smoothed_derivative_detector(steps: np.ndarray, values: np.ndarray, window: int = 5, threshold: float = 0.0) -> Optional[float]:
    """
    Detect changepoint using a smoothed derivative.
    Returns the step with the maximum derivative that exceeds the threshold.
    """
    if len(steps) < window:
        return None

    # Moving average smoothing
    kernel = np.ones(window) / window
    smoothed_values = np.convolve(values, kernel, mode='valid')
    smoothed_steps = steps[window - 1:]

    if len(smoothed_steps) < 2:
        return None

    # Discrete derivative
    dx = np.diff(smoothed_steps)
    dx[dx == 0] = 1e-9  # prevent division by zero
    dy = np.diff(smoothed_values)
    derivative = dy / dx

    max_idx = np.argmax(derivative)
    if derivative[max_idx] >= threshold:
        # Return the step corresponding to the interval where derivative is max
        return float(smoothed_steps[max_idx + 1])

    return None

def bocd_detector(steps: np.ndarray, values: np.ndarray, hazard: float = 1/100,
                 mean0: float = 0.5, var0: float = 1.0, varx: float = 0.01) -> Optional[float]:
    """
    Bayesian Online Changepoint Detection (BOCD) with a Gaussian observation model.
    """
    T = len(values)
    if T < 2:
        return None

    # Initialize run length probabilities
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Sufficient statistics for each run length
    # Mean of Gaussian prior
    mu = np.array([mean0])
    # Precision of Gaussian prior
    kappa = np.array([1.0 / var0])

    max_run_lengths = []

    for t in range(1, T + 1):
        x = values[t - 1]

        # Evaluate predictive probability for current observation given each run length
        # Predictive distribution is Gaussian with mean mu and variance (1/kappa) + varx
        pred_var = (1.0 / kappa) + varx
        pred_prob = np.exp(-0.5 * ((x - mu) ** 2) / pred_var) / np.sqrt(2 * np.pi * pred_var)

        # Calculate growth probabilities
        R[t, 1:t+1] = R[t-1, 0:t] * pred_prob * (1 - hazard)

        # Calculate changepoint probability
        R[t, 0] = np.sum(R[t-1, 0:t] * pred_prob * hazard)

        # Normalize
        R[t, 0:t+1] /= np.sum(R[t, 0:t+1])

        # Update sufficient statistics
        # Append prior for new run length
        mu = np.append(mean0, mu)
        kappa = np.append(1.0 / var0, kappa)

        # Update for previous run lengths
        kappa[1:] += 1.0 / varx
        mu[1:] = (mu[1:] * (kappa[1:] - 1.0 / varx) + x / varx) / kappa[1:]

        max_run_lengths.append(np.argmax(R[t, :t]))

    # Find the most likely changepoint by looking backward
    # We find the step where the run length drops to 0
    # Actually, a simpler robust way to extract the changepoint from the MAP run lengths:
    cps = []
    current_rl = max_run_lengths[-1]
    for t in reversed(range(1, T)):
        if max_run_lengths[t-1] == 0 and current_rl > 0:
            cps.append(t)
        current_rl = max_run_lengths[t-1]

    if not cps:
        # alternatively, track where run length drops maximally
        rl_diff = np.diff(max_run_lengths)
        if len(rl_diff) > 0 and np.min(rl_diff) < -1:
            return float(steps[np.argmin(rl_diff) + 1])
        return None

    # Return the first changepoint found (closest to start)
    return float(steps[min(cps)])


def bootstrap_ci(steps: np.ndarray, values: np.ndarray, detector_fn: Callable,
                n_bootstraps: int = 100, ci: float = 0.95) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute a confidence interval for a changepoint using bootstrap resampling with jitter.
    Returns (median_cp, lower_ci, upper_ci).
    """
    if len(steps) < 4:
        return None, None, None

    # Get base estimate
    base_cp = detector_fn(steps, values)
    if base_cp is None:
        return None, None, None

    bootstrapped_cps = []
    n = len(steps)

    for _ in range(n_bootstraps):
        # Resample indices with replacement
        idx = np.sort(np.random.choice(n, n, replace=True))

        boot_steps = steps[idx].astype(float).copy()
        boot_values = values[idx].copy()

        # Add small jitter to steps to avoid duplicate steps breaking assumptions
        # Jitter magnitude should be smaller than min step difference
        if n > 1:
            diffs = np.diff(np.unique(steps))
            min_diff = np.min(diffs) if len(diffs) > 0 else 1.0
            jitter = np.random.uniform(-0.1 * min_diff, 0.1 * min_diff, size=n)
            boot_steps += jitter

        # Ensure monotonic increasing steps after jitter
        boot_steps = np.sort(boot_steps)
        # We need to sort values based on the sorted jittered steps too
        sort_idx = np.argsort(boot_steps)
        boot_steps = boot_steps[sort_idx]
        boot_values = boot_values[sort_idx]

        cp = detector_fn(boot_steps, boot_values)
        if cp is not None:
            bootstrapped_cps.append(cp)

    if not bootstrapped_cps:
        return base_cp, base_cp, base_cp

    bootstrapped_cps = np.array(bootstrapped_cps)
    alpha = 1.0 - ci
    lower = np.percentile(bootstrapped_cps, alpha / 2 * 100)
    upper = np.percentile(bootstrapped_cps, (1 - alpha / 2) * 100)

    return base_cp, float(lower), float(upper)
