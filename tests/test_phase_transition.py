import pytest
import numpy as np
from src.phase_transition import (
    detect_grokking_step,
    detect_train_memorization_step,
    calculate_grokking_ratio,
    detect_weight_norm_rupture,
    analyze_experiment
)

def test_detect_grokking_step():
    # Never groks
    accs = [0.1, 0.2, 0.3, 0.8, 0.85, 0.89]
    assert detect_grokking_step(accs) == -1

    # Groks at index 4 (0.91)
    accs = [0.1, 0.2, 0.3, 0.8, 0.91, 0.95, 0.99]
    assert detect_grokking_step(accs) == 4

    # Groks but dips (min_sustained_steps)
    accs = [0.1, 0.95, 0.8, 0.95, 0.95, 0.95]
    # With min_sustained_steps=1 it returns 1
    assert detect_grokking_step(accs, min_sustained_steps=1) == 1
    # With min_sustained_steps=3 it returns 3
    assert detect_grokking_step(accs, min_sustained_steps=3) == 3

def test_calculate_grokking_ratio():
    train_accs = [0.1, 0.5, 0.95, 0.99, 1.0, 1.0]
    test_accs = [0.1, 0.1, 0.2, 0.5, 0.92, 0.99]

    # Train reaches 0.90 at index 2
    # Test reaches 0.90 at index 4
    # Ratio = 4 / 2 = 2.0
    ratio = calculate_grokking_ratio(train_accs, test_accs)
    assert np.isclose(ratio, 2.0)

    # No grokking
    test_accs_no_grok = [0.1] * 6
    assert calculate_grokking_ratio(train_accs, test_accs_no_grok) == -1.0

def test_detect_weight_norm_rupture():
    # Two linear segments: y = 2x up to x=3, then y = 6 - 0.5(x-3)
    # x: 0, 1, 2, 3, 4, 5, 6
    # y: 0, 2, 4, 6, 5.5, 5, 4.5
    norms = [0, 2, 4, 6, 5.5, 5, 4.5]
    rupture = detect_weight_norm_rupture(norms)
    # The rupture should be detected around index 3 or 4
    assert rupture in [3, 4]

def test_analyze_experiment():
    history = [
        {"step": 0, "train_acc": 0.1, "test_acc": 0.1, "weight_norm": 10.0},
        {"step": 100, "train_acc": 0.95, "test_acc": 0.2, "weight_norm": 20.0},
        {"step": 200, "train_acc": 0.99, "test_acc": 0.3, "weight_norm": 30.0},
        {"step": 300, "train_acc": 1.0, "test_acc": 0.95, "weight_norm": 25.0},
        {"step": 400, "train_acc": 1.0, "test_acc": 0.98, "weight_norm": 20.0},
    ]

    res = analyze_experiment(history)
    assert res["memorization_step"] == 100
    assert res["grokking_step"] == 300
    assert np.isclose(res["grokking_ratio"], 3.0)
    assert res["weight_norm_rupture_step"] in [200, 300]
