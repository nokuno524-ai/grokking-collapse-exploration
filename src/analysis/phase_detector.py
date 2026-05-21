"""
Phase transition detection utilities for grokking analysis.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


def detect_grokking_transition(
    metrics: List[float],
    steps: List[int],
    threshold: float = 0.95,
    window_size: int = 50
) -> Optional[Dict[str, Any]]:
    """
    Detect the exact step where grokking occurs.
    Defined as the first step where the metric (e.g. test accuracy)
    exceeds `threshold` and remains above it for at least `window_size` steps.

    Args:
        metrics: List of metric values (e.g., test accuracy).
        steps: List of corresponding training steps.
        threshold: The threshold to cross for grokking.
        window_size: How many subsequent steps must remain above threshold.

    Returns:
        A dictionary with the grokking step and confidence metrics, or None if not found.
    """
    if not metrics or len(metrics) != len(steps):
        raise ValueError("metrics and steps must be non-empty and of same length")

    if len(metrics) < window_size:
        return None

    for i in range(len(metrics) - window_size + 1):
        window_metrics = metrics[i:i+window_size]
        if all(m >= threshold for m in window_metrics):
            # Calculate confidence metrics
            mean_val = np.mean(window_metrics)
            std_val = np.std(window_metrics)

            # Find the true drop-off stability
            post_transition = metrics[i:]
            stability_score = sum(1 for m in post_transition if m >= threshold) / len(post_transition)

            return {
                "step": steps[i],
                "index": i,
                "confidence_metrics": {
                    "window_mean": float(mean_val),
                    "window_std": float(std_val),
                    "stability_score": float(stability_score)
                }
            }

    return None


def detect_fourier_shift(
    fourier_concentration: List[float],
    steps: List[int],
    uniform_threshold: float = 0.1,
    concentrated_threshold: float = 0.5,
    window_size: int = 10
) -> Dict[str, Any]:
    """
    Detect the phase shift when Fourier concentration moves from uniform to concentrated.

    Args:
        fourier_concentration: List of Fourier concentration values.
        steps: List of corresponding training steps.
        uniform_threshold: Value below which concentration is considered 'uniform' or random.
        concentrated_threshold: Value above which concentration is considered 'concentrated'.
        window_size: Smoothing window to ensure shift is stable.

    Returns:
        Dict containing shift details: start_step, end_step, is_shifted, and confidence metrics.
    """
    if not fourier_concentration or len(fourier_concentration) != len(steps):
        raise ValueError("fourier_concentration and steps must be non-empty and of same length")

    if len(fourier_concentration) < window_size * 2:
        return {"is_shifted": False, "start_step": None, "end_step": None, "confidence_metrics": {}}

    # Find last point where it was consistently 'uniform'
    uniform_end_idx = None
    for i in range(len(fourier_concentration) - window_size):
        if all(m <= uniform_threshold for m in fourier_concentration[i:i+window_size]):
            uniform_end_idx = i + window_size - 1

    # Find first point where it becomes consistently 'concentrated'
    concentrated_start_idx = None
    for i in range(len(fourier_concentration) - window_size + 1):
        if all(m >= concentrated_threshold for m in fourier_concentration[i:i+window_size]):
            concentrated_start_idx = i
            break

    if uniform_end_idx is not None and concentrated_start_idx is not None and uniform_end_idx < concentrated_start_idx:
        # Calculate confidence metrics
        shift_duration_steps = steps[concentrated_start_idx] - steps[uniform_end_idx]

        # Gradient approximation during shift
        shift_gradient = (fourier_concentration[concentrated_start_idx] - fourier_concentration[uniform_end_idx]) / (concentrated_start_idx - uniform_end_idx)

        return {
            "is_shifted": True,
            "start_step": steps[uniform_end_idx],
            "end_step": steps[concentrated_start_idx],
            "confidence_metrics": {
                "shift_duration_steps": shift_duration_steps,
                "shift_gradient": float(shift_gradient)
            }
        }

    return {"is_shifted": False, "start_step": None, "end_step": None, "confidence_metrics": {}}
