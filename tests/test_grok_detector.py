from typing import List, Dict
import numpy as np
import pytest

from src.analysis.grok_detector import detect_grok_step, compute_grok_ci, is_never_grok

def generate_mock_history(
    total_steps: int = 100,
    grok_step_idx: int = -1,
    final_acc: float = 1.0,
    plateau: bool = False,
    noise_level: float = 0.01,
) -> List[Dict]:
    """Helper to create synthetic training histories."""
    history = []

    for i in range(total_steps):
        step = (i + 1) * 10

        # Base accuracy
        if plateau:
            acc = final_acc
        else:
            if grok_step_idx != -1 and i >= grok_step_idx:
                acc = final_acc
            elif grok_step_idx != -1:
                # Pre-grok is random chance
                acc = 0.02
            else:
                # Slow monotonic non-grok
                acc = 0.02 + (final_acc - 0.02) * (i / total_steps)

        # Add noise
        acc = acc + np.random.normal(0, noise_level)
        acc = max(0.0, min(1.0, acc)) # clamp

        history.append({
            "step": step,
            "test_acc": acc,
        })

    return history


def test_detect_grok_step_sharp_jump():
    """Test detector correctly identifies a planted sharp jump."""
    np.random.seed(42)

    # Sharp jump at index 60 (step 610)
    history = generate_mock_history(total_steps=100, grok_step_idx=60, final_acc=1.0, noise_level=0.01)

    step = detect_grok_step(history, window=10)
    # The jump happens between index 60 and previous window, so it should report the step around index 60.
    # index 60 -> step 610
    assert step is not None
    assert 600 <= step <= 670  # widened range since max jump could be reported slightly later in the window depending on noise

def test_detect_grok_step_noisy_plateau():
    """Test detector does not fire on a noisy plateau."""
    np.random.seed(42)
    history = generate_mock_history(total_steps=100, plateau=True, final_acc=0.5, noise_level=0.05)

    step = detect_grok_step(history, window=10)
    assert step is None

def test_detect_grok_step_slow_monotonic():
    """Test detector does not identify a non-grok monotonic curve."""
    np.random.seed(42)
    history = generate_mock_history(total_steps=100, final_acc=0.8, noise_level=0.01)

    step = detect_grok_step(history, window=10)
    assert step is None

def test_compute_grok_ci():
    """Test CI returns sensible bounds."""
    np.random.seed(42)
    history = generate_mock_history(total_steps=100, grok_step_idx=60, final_acc=1.0, noise_level=0.01)

    lower, upper = compute_grok_ci(history, window=10, n_bootstraps=100)

    assert lower is not None
    assert upper is not None
    assert lower <= 610
    assert upper >= 610

def test_is_never_grok():
    """Test never-grok classification."""
    np.random.seed(42)

    # Plateau at 0.5
    history_plateau = generate_mock_history(total_steps=100, plateau=True, final_acc=0.5, noise_level=0.001)
    assert bool(is_never_grok(history_plateau, threshold=0.95, epsilon=1e-3)) is True

    # Grok curve
    history_grok = generate_mock_history(total_steps=100, grok_step_idx=60, final_acc=1.0, noise_level=0.01)
    assert bool(is_never_grok(history_grok, threshold=0.95, epsilon=1e-3)) is False
