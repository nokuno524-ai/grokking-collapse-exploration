import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, List, Optional

def detect_grokking(test_accs: List[float], steps: List[int], window_length: int = 15, polyorder: int = 2) -> Tuple[bool, Optional[int]]:
    """
    Detect grokking point by finding where the test accuracy sharply increases.
    Uses Savitzky-Golay filter to smooth the curve and finds the maximum of the second derivative.

    Args:
        test_accs: List of test accuracies over time.
        steps: List of corresponding step numbers.
        window_length: Window length for the savgol filter.
        polyorder: Polynomial order for the savgol filter.

    Returns:
        Tuple of (grokked_bool, grokking_step_if_any)
    """
    if len(test_accs) < 3:
        return False, None

    # Strict fallbacks for short sequences
    if len(test_accs) < window_length:
        window_length = len(test_accs)
        if window_length % 2 == 0:
            window_length -= 1

    if window_length < 3:
        # Fallback to simple threshold if sequence is too short for any filtering
        for acc, step in zip(test_accs, steps):
            if acc > 0.9:
                return True, step
        return False, None

    if polyorder >= window_length:
        polyorder = window_length - 1

    # Ensure arrays
    accs = np.array(test_accs)

    # Smooth test accuracies
    smoothed = savgol_filter(accs, window_length=window_length, polyorder=polyorder)

    # Calculate first and second derivatives
    dy = np.gradient(smoothed)
    ddy = np.gradient(dy)

    # Find the peak of the second derivative (maximum acceleration in accuracy)
    # This corresponds to the inflection point at the start of the "S-curve" or sharp jump
    peak_idx = np.argmax(ddy)

    # We also require that the final accuracy actually reached a good level to count as grokking
    if accs[-1] > 0.9 and np.max(dy) > 0.01: # simple heuristics for jump and success
        return True, steps[peak_idx]

    return False, None

if __name__ == "__main__":
    # Test
    accs = [0.1]*10 + [0.2, 0.5, 0.8, 0.95] + [0.98]*5
    steps = list(range(len(accs)))
    print(detect_grokking(accs, steps))
