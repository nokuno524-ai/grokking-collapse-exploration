import numpy as np
from src.analysis.grok_detector.detectors import (
    piecewise_linear_detector,
    smoothed_derivative_detector,
    bocd_detector,
    bootstrap_ci
)

def test_piecewise_linear_detector():
    steps = np.arange(10)
    # create a step function at x=5
    values = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    cp = piecewise_linear_detector(steps, values)
    assert cp is not None
    # Depending on SSE fit, it should find breakpoint around 4 or 5
    assert 4 <= cp <= 6

def test_smoothed_derivative_detector():
    steps = np.arange(20)
    # Smooth step transition at x=10
    values = 1 / (1 + np.exp(-(steps - 10)))

    cp = smoothed_derivative_detector(steps, values, window=3)
    assert cp is not None
    assert 8 <= cp <= 12

def test_bocd_detector():
    np.random.seed(42)
    steps = np.arange(50)
    # Mean shift at x=25
    values = np.concatenate([np.random.normal(0, 0.01, 25), np.random.normal(5, 0.01, 25)])

    # Provide generous priors for the test
    cp = bocd_detector(steps, values, mean0=2.5, var0=10.0, varx=0.01)
    assert cp is not None
    assert 20 <= cp <= 30

def test_bootstrap_ci():
    steps = np.arange(20)
    values = np.concatenate([np.zeros(10), np.ones(10)])

    cp, lower, upper = bootstrap_ci(steps, values, piecewise_linear_detector, n_bootstraps=20)
    assert cp is not None
    assert lower is not None
    assert upper is not None
    assert lower <= cp <= upper
