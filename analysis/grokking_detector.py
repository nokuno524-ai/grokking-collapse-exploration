"""
Grokking Detector.
Automatically detects the grokking point via the second derivative of the accuracy curve,
computes the grokking gap, detects training phases, and outputs phase boundaries.
"""

import numpy as np
from scipy.signal import savgol_filter

def extract_phases(accuracy_curve: np.ndarray, steps: np.ndarray, window_length: int = 11, polyorder: int = 3):
    """
    Extract training phases from the accuracy curve.

    Phases:
    1. Memorization: Train acc goes up, test acc is low.
    2. Grokking (Circuit formation): Test acc sharply increases.
    3. Generalization: Test acc is high and stable.

    Args:
        accuracy_curve: Array of test accuracy values.
        steps: Array of training steps corresponding to the accuracies.
        window_length: Length of the savgol filter window. Must be odd.
        polyorder: Order of the polynomial for the savgol filter.

    Returns:
        dict: Phase boundaries and grokking step.
    """
    if len(accuracy_curve) < window_length or window_length < 3:
        # Not enough points, fallback
        if len(accuracy_curve) > 0:
            idx = np.argmax(accuracy_curve >= 0.95)
            if accuracy_curve[idx] >= 0.95:
                return {"grokking_step": steps[idx], "grokking_gap": 0.0, "phases": {}}
        return {"grokking_step": None, "grokking_gap": 0.0, "phases": {}}

    # Smooth the curve
    smoothed = savgol_filter(accuracy_curve, window_length=window_length, polyorder=polyorder)

    # First derivative (rate of change)
    first_deriv = np.gradient(smoothed)

    # Second derivative (acceleration)
    second_deriv = np.gradient(first_deriv)

    # Grokking point is typically where acceleration is maximal (start of the S-curve's upward swing)
    max_accel_idx = np.argmax(second_deriv)

    # Verify it actually groks
    if np.max(smoothed) < 0.8:
        return {"grokking_step": None, "grokking_gap": 0.0, "phases": {}}

    grok_step = steps[max_accel_idx]

    # Generalization phase starts when first derivative drops near zero after grokking
    post_grok_deriv = first_deriv[max_accel_idx:]
    gen_idx_offset = np.argmax(post_grok_deriv < 0.01)
    if post_grok_deriv[gen_idx_offset] < 0.01:
        gen_idx = max_accel_idx + gen_idx_offset
    else:
        gen_idx = len(steps) - 1

    # Grokking gap: diff between train acc and test acc? We need train acc for that.
    # Here we define the gap visually from the curve jump.
    grokking_gap = smoothed[gen_idx] - smoothed[max_accel_idx]

    phases = {
        "memorization_end": grok_step,
        "grokking_start": grok_step,
        "grokking_end": steps[gen_idx],
        "generalization_start": steps[gen_idx]
    }

    return {
        "grokking_step": grok_step,
        "grokking_gap": float(grokking_gap),
        "phases": phases
    }

def analyze_training_run(results_history: list) -> dict:
    """
    Analyze a training run from its history to extract grokking point.
    """
    steps = np.array([entry['step'] for entry in results_history])
    test_acc = np.array([entry['test_acc'] for entry in results_history])

    if len(steps) == 0:
        return {}

    window_length = min(11, len(steps))
    if window_length % 2 == 0:
        window_length -= 1

    if window_length < 3:
        return extract_phases(test_acc, steps, window_length=0) # Fallback triggered

    return extract_phases(test_acc, steps, window_length=window_length)
