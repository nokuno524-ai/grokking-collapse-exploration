import json
import numpy as np
from typing import List, Dict, Union, Any

def detect_grokking_step(test_accuracies: List[float], threshold: float = 0.90, min_sustained_steps: int = 1) -> int:
    """
    Analyzes training curves to automatically detect grokking onset (test accuracy >= threshold sustained).
    Returns -1 if not found.
    """
    if not test_accuracies:
        return -1

    # Needs to be sustained for min_sustained_steps
    for i in range(len(test_accuracies) - min_sustained_steps + 1):
        if all(acc >= threshold for acc in test_accuracies[i:i+min_sustained_steps]):
            return i

    # If min_sustained_steps is longer than list, just check if the last few meet it
    if len(test_accuracies) < min_sustained_steps:
         if all(acc >= threshold for acc in test_accuracies):
             return 0

    return -1

def detect_train_memorization_step(train_accuracies: List[float], threshold: float = 0.90, min_sustained_steps: int = 1) -> int:
    """
    Finds when training accuracy hits threshold.
    """
    return detect_grokking_step(train_accuracies, threshold, min_sustained_steps)

def calculate_grokking_ratio(train_accuracies: List[float], test_accuracies: List[float], threshold: float = 0.90) -> float:
    """
    Computes the "grokking ratio" (grokking_step / training_steps_to_90pct_train_acc).
    Returns -1.0 if memorization or grokking did not occur.
    """
    train_step = detect_train_memorization_step(train_accuracies, threshold)
    grok_step = detect_grokking_step(test_accuracies, threshold)

    if train_step == -1 or grok_step == -1:
        return -1.0

    # Prevent division by zero
    if train_step == 0:
        return float('inf') if grok_step > 0 else 1.0

    return grok_step / train_step

def detect_weight_norm_rupture(weight_norms: List[float]) -> int:
    """
    Detects phase transitions in weight norms using change-point detection.
    Uses exhaustive search piecewise linear fit.
    Returns the index of the detected rupture/transition point, or -1 if < 3 points.
    """
    n = len(weight_norms)
    if n < 3:
        return -1

    y = np.array(weight_norms)
    x = np.arange(n)

    best_error = float('inf')
    best_idx = -1

    # We need at least 2 points for each segment to fit a line
    for i in range(2, n - 2):
        x1, y1 = x[:i], y[:i]
        x2, y2 = x[i:], y[i:]

        # Fit lines
        c1 = np.polyfit(x1, y1, 1)
        c2 = np.polyfit(x2, y2, 1)

        # Compute errors
        err1 = np.sum((np.polyval(c1, x1) - y1) ** 2)
        err2 = np.sum((np.polyval(c2, x2) - y2) ** 2)
        total_err = err1 + err2

        if total_err < best_error:
            best_error = total_err
            best_idx = i

    return best_idx

def analyze_experiment(history: List[Dict[str, Any]], step_key: str = 'step') -> Dict[str, Any]:
    """
    Analyzes an experiment's history and outputs a structured dict with transition points.
    """
    if not history:
        return {}

    train_accs = [entry.get('train_acc', 0.0) for entry in history]
    test_accs = [entry.get('test_acc', 0.0) for entry in history]
    weight_norms = [entry.get('weight_norm', 0.0) for entry in history]

    grok_idx = detect_grokking_step(test_accs)
    mem_idx = detect_train_memorization_step(train_accs)
    rupture_idx = detect_weight_norm_rupture(weight_norms)

    # Map indices back to actual steps if available
    grok_step = history[grok_idx][step_key] if grok_idx != -1 else -1
    mem_step = history[mem_idx][step_key] if mem_idx != -1 else -1
    rupture_step = history[rupture_idx][step_key] if rupture_idx != -1 else -1

    grok_ratio = calculate_grokking_ratio(train_accs, test_accs)
    if grok_step != -1 and mem_step != -1 and mem_step != 0:
        # Re-compute ratio using actual step values, unless you want it based on index
        # Let's use the actual step values
        grok_ratio = grok_step / mem_step

    return {
        "grokking_step": grok_step,
        "memorization_step": mem_step,
        "grokking_ratio": grok_ratio,
        "weight_norm_rupture_step": rupture_step
    }
