import numpy as np
import pytest
from src.analysis.grok_detector.detectors import (
    threshold_detector,
    logistic_detector,
    binary_segmentation_detector,
    derivative_maximum_detector,
    bootstrap_ci
)

def test_threshold_detector():
    steps = np.arange(10)
    accs = np.array([0.1, 0.1, 0.1, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96])
    assert threshold_detector(steps, accs, threshold=0.95, dwell=1) == 3.0
    assert threshold_detector(steps, accs, threshold=0.95, dwell=3) == 3.0

    accs_no_grok = np.ones(10) * 0.5
    assert threshold_detector(steps, accs_no_grok) is None

    accs_blip = np.array([0.1, 0.1, 0.96, 0.1, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96])
    assert threshold_detector(steps, accs_blip, threshold=0.95, dwell=2) == 4.0

def test_logistic_detector():
    steps = np.linspace(0, 1000, 100)
    # create a perfect logistic curve that crosses 0.95 at x=500
    L = 1.0
    k = 0.05
    x0 = 400
    # L / (1 + exp(-k(x-x0)))
    accs = L / (1 + np.exp(-k * (steps - x0)))

    cross_step = logistic_detector(steps, accs, threshold=0.95)
    assert cross_step is not None

    # check that at cross_step, the value is roughly 0.95
    val_at_cross = L / (1 + np.exp(-k * (cross_step - x0)))
    assert np.isclose(val_at_cross, 0.95, atol=1e-2)

    # Flat line should fail
    assert logistic_detector(steps, np.ones_like(steps)*0.5) is None

def test_binary_segmentation_detector():
    steps = np.arange(10)
    accs = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    cp = binary_segmentation_detector(steps, accs)
    assert cp == 4.0

def test_derivative_maximum_detector():
    steps = np.arange(10)
    accs = np.array([0.1, 0.1, 0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    # steepest jump is around step 3-4
    cp = derivative_maximum_detector(steps, accs, window=3)
    assert cp is not None

def test_bootstrap_ci():
    steps = np.arange(20)
    accs = np.concatenate([np.ones(10)*0.1, np.ones(10)*0.98])

    base_val, lower, upper = bootstrap_ci(steps, accs, threshold_detector, n_bootstraps=50)
    assert base_val == 10.0
    assert lower is not None
    assert upper is not None
    assert lower <= base_val <= upper
