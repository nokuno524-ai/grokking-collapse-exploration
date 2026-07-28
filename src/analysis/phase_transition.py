import numpy as np
from scipy.optimize import curve_fit

def detect_grokking_point(test_acc_history, threshold=0.9):
    """
    Detect the exact grokking point based on the discrete derivative of test accuracy.
    Finds the step where test accuracy crosses the threshold, returning -1 if max accuracy < 0.9.

    Args:
        test_acc_history: List or array of test accuracies.
        threshold: The accuracy threshold to consider as 'grokked'.

    Returns:
        Index of the grokking step in the provided array, or -1 if threshold not met.
    """
    test_acc_history = np.array(test_acc_history)

    if len(test_acc_history) == 0 or np.max(test_acc_history) < threshold:
        return -1

    diff = np.diff(test_acc_history)
    # the grokking point is often characterized by a sharp rise.
    # To keep it simple, we just find the first point where acc >= threshold
    idx = np.argmax(test_acc_history >= threshold)
    return int(idx)

def _logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def measure_phase_transition_sharpness(test_acc_history, steps=None):
    """
    Measure the sharpness of the phase transition by fitting a logistic curve.

    Args:
        test_acc_history: List or array of test accuracies.
        steps: List or array of step numbers corresponding to accuracies.

    Returns:
        dict: Parameters of the logistic curve fitting {'L', 'k', 'x0'}.
    """
    test_acc_history = np.array(test_acc_history)
    if steps is None:
        steps = np.arange(len(test_acc_history))
    else:
        steps = np.array(steps)

    if len(test_acc_history) < 4:
        # Not enough data points to fit logistic
        return {'L': np.nan, 'k': np.nan, 'x0': np.nan}

    try:
        # Initial guess: L=max(y), x0=median(x), k=1
        p0 = [np.max(test_acc_history), np.median(steps), 1.0]
        # bounds to keep L near 1, k positive
        bounds = ([0.0, 0.0, -np.inf], [2.0, np.inf, np.inf])

        # Normalize x to make fitting stable
        x_norm = steps / np.max(steps) if np.max(steps) > 0 else steps
        p0_norm = [np.max(test_acc_history), np.median(x_norm), 10.0]

        popt, _ = curve_fit(_logistic, x_norm, test_acc_history, p0=p0_norm, bounds=bounds, maxfev=10000)

        L_norm, x0_norm, k_norm = popt

        # Scale back
        max_step = np.max(steps) if np.max(steps) > 0 else 1.0

        L = L_norm
        x0 = x0_norm * max_step
        k = k_norm / max_step

        return {'L': L, 'k': k, 'x0': x0}
    except Exception as e:
        return {'L': np.nan, 'k': np.nan, 'x0': np.nan}


def compute_grokking_delay(train_acc, test_acc, steps=None, train_threshold=0.9, test_threshold=0.9):
    """
    Compute the delay between memorization (train acc) and generalization (test acc).

    Args:
        train_acc: Array of train accuracies.
        test_acc: Array of test accuracies.
        steps: Array of step numbers.
        train_threshold: Threshold for train accuracy.
        test_threshold: Threshold for test accuracy.

    Returns:
        Integer representing the gap in steps. Returns -1 if either condition is not met.
    """
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)

    if steps is None:
        steps = np.arange(len(train_acc))
    else:
        steps = np.array(steps)

    train_idx = np.argmax(train_acc >= train_threshold)
    test_idx = np.argmax(test_acc >= test_threshold)

    if np.max(train_acc) < train_threshold or np.max(test_acc) < test_threshold:
        return -1

    return int(steps[test_idx] - steps[train_idx])


def bootstrap_grokking_ci(grokking_steps, num_bootstraps=1000, ci=0.95):
    """
    Compute bootstrap confidence intervals for the mean grokking step.

    Args:
        grokking_steps: List of grokking steps from multiple runs.
        num_bootstraps: Number of bootstrap iterations.
        ci: Confidence interval (e.g. 0.95).

    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    grokking_steps = np.array(grokking_steps)
    grokking_steps = grokking_steps[grokking_steps > 0] # Filter out failures

    if len(grokking_steps) == 0:
        return (np.nan, np.nan, np.nan)

    means = []
    for _ in range(num_bootstraps):
        sample = np.random.choice(grokking_steps, size=len(grokking_steps), replace=True)
        means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)

    return (np.mean(grokking_steps), lower, upper)
