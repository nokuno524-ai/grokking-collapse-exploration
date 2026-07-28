import numpy as np
import pytest
from src.analysis.phase_transition import (
    detect_grokking_point,
    measure_phase_transition_sharpness,
    compute_grokking_delay,
    bootstrap_grokking_ci
)

def test_detect_grokking_point():
    # Acc never reaches threshold
    acc = [0.1, 0.2, 0.3, 0.4]
    assert detect_grokking_point(acc, threshold=0.9) == -1

    # Acc reaches threshold at index 3
    acc = [0.1, 0.2, 0.8, 0.95, 0.96]
    assert detect_grokking_point(acc, threshold=0.9) == 3

def test_measure_phase_transition_sharpness():
    # Not enough points
    acc = [0.1, 0.2]
    res = measure_phase_transition_sharpness(acc)
    assert np.isnan(res['k'])

    # Perfect logistic
    x = np.arange(100)
    L = 1.0
    k = 0.5
    x0 = 50
    y = L / (1 + np.exp(-k * (x - x0)))

    res = measure_phase_transition_sharpness(y, x)
    assert not np.isnan(res['k'])
    assert np.isclose(res['L'], L, atol=0.1)

def test_compute_grokking_delay():
    train_acc = [0.5, 0.95, 0.95, 0.95, 0.95]
    test_acc = [0.2, 0.3, 0.4, 0.95, 0.96]
    steps = [0, 100, 200, 300, 400]

    # train hits at idx 1 (100), test hits at idx 3 (300) => diff 200
    assert compute_grokking_delay(train_acc, test_acc, steps) == 200

    # Test never hits
    test_acc = [0.2, 0.3, 0.4, 0.5, 0.6]
    assert compute_grokking_delay(train_acc, test_acc, steps) == -1

def test_bootstrap_grokking_ci():
    # Empty after filter
    steps = [-1, -1, 0]
    mean, lower, upper = bootstrap_grokking_ci(steps)
    assert np.isnan(mean)

    # Valid steps
    steps = [1000, 1100, 900, 1050, 950]
    mean, lower, upper = bootstrap_grokking_ci(steps, num_bootstraps=100)
    assert np.isclose(mean, 1000)
    assert lower <= mean <= upper
