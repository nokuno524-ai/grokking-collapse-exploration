from typing import List, Dict, Optional, Tuple
import numpy as np

def detect_grok_step(history: List[Dict], window: int = 10) -> Optional[int]:
    """
    Find the step with the maximum test_acc jump within a moving window.
    Only returns a valid step if the final test_acc crosses 0.95.
    """
    if not history or len(history) < window:
        return None

    # Needs to cross the grokking threshold
    if history[-1].get("test_acc", 0.0) < 0.95:
        return None

    max_jump = -1.0
    grok_step = None

    for i in range(window, len(history)):
        past_acc = history[i - window].get("test_acc", 0.0)
        curr_acc = history[i].get("test_acc", 0.0)
        jump = curr_acc - past_acc
        if jump > max_jump:
            max_jump = jump
            grok_step = history[i]["step"]

    return grok_step

def compute_grok_ci(history: List[Dict], window: int = 10, n_bootstraps: int = 1000) -> Tuple[Optional[int], Optional[int]]:
    """
    Compute a 95% confidence interval for the grok step using bootstrap resampling
    of the accuracy jumps (simulating noise over the trajectory).
    Returns (lower_bound, upper_bound)
    """
    if not history or len(history) < window:
        return None, None

    if history[-1].get("test_acc", 0.0) < 0.95:
        return None, None

    # We construct a sequence of jumps
    jumps = []
    steps = []
    for i in range(window, len(history)):
        past_acc = history[i - window].get("test_acc", 0.0)
        curr_acc = history[i].get("test_acc", 0.0)
        jumps.append(curr_acc - past_acc)
        steps.append(history[i]["step"])

    if not jumps:
        return None, None

    jumps_arr = np.array(jumps)
    steps_arr = np.array(steps)

    bootstrapped_steps = []
    for _ in range(n_bootstraps):
        # We add gaussian noise based on the standard deviation of jumps.
        noise = np.random.normal(0, np.std(jumps_arr) + 1e-6, size=len(jumps_arr))
        noisy_jumps = jumps_arr + noise
        max_idx = np.argmax(noisy_jumps)
        bootstrapped_steps.append(steps_arr[max_idx])

    lower = int(np.percentile(bootstrapped_steps, 2.5))
    upper = int(np.percentile(bootstrapped_steps, 97.5))
    return lower, upper


def is_never_grok(history: List[Dict], threshold: float = 0.95, window: int = 10, epsilon: float = 1e-4) -> bool:
    """
    Check if the model never groks: final accuracy is below threshold,
    and it has plateaued (variance in last `window` steps < epsilon).
    """
    if not history or len(history) < window:
        return False

    final_acc = history[-1].get("test_acc", 0.0)
    if final_acc >= threshold:
        return False

    last_window_accs = [e.get("test_acc", 0.0) for e in history[-window:]]
    variance = np.var(last_window_accs)

    return variance < epsilon
