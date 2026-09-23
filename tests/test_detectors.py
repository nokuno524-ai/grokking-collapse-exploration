import pytest
import numpy as np
from src.analysis.grok_detector.detectors import (
    plateau_detector,
    max_2nd_derivative_detector,
    fit_sigmoid_transition
)

def test_plateau_detector():
    steps = np.arange(10)
    # Reaches 0.95 at step 5 and stays there
    accs = np.array([0.1, 0.2, 0.4, 0.8, 0.9, 0.96, 0.97, 0.98, 0.99, 1.0])

    cliff = plateau_detector(steps, accs, threshold=0.95, window=1)
    assert cliff == 5

    # Never reaches threshold
    accs_low = np.array([0.1]*10)
    assert plateau_detector(steps, accs_low, threshold=0.95, window=1) is None

    # Drops below threshold again
    accs_drop = np.array([0.1, 0.96, 0.97, 0.9, 0.96, 0.97, 0.98, 0.99, 1.0, 1.0])
    assert plateau_detector(steps, accs_drop, threshold=0.95, window=1) == 4

def test_max_2nd_derivative_detector():
    steps = np.linspace(0, 100, 101)
    # Sigmoid transition centered at 50
    accs = 1 / (1 + np.exp(-0.2 * (steps - 50)))

    # Max 2nd derivative (acceleration) for sigmoid occurs earlier than the midpoint
    cliff = max_2nd_derivative_detector(steps, accs, window=1)
    # The max second derivative for a sigmoid 1/(1+e^(-k(x-x0))) is around x0 - 1.317/k
    # Here k=0.2, x0=50. x0 - 1.317/0.2 = 50 - 6.585 = 43.415
    assert 42 <= cliff <= 45

def test_fit_sigmoid_transition():
    steps = np.linspace(0, 100, 101)
    # Perfect sigmoid transition at exactly 42
    accs = 0.9 / (1 + np.exp(-0.15 * (steps - 42))) + 0.05

    cliff, params = fit_sigmoid_transition(steps, accs)

    assert cliff is not None
    assert abs(cliff - 42) <= 1
    assert abs(params['x0'] - 42) < 1.0
    assert abs(params['L'] - 0.9) < 0.1
    assert abs(params['b'] - 0.05) < 0.1

    # Test with flat line (no transition)
    accs_flat = np.full(101, 0.5)
    cliff_flat, params_flat = fit_sigmoid_transition(steps, accs_flat)
    assert cliff_flat is None
    assert "reason" in params_flat
