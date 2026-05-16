import numpy as np
import pytest
from src.phase_detector import PhaseTransitionDetector

def test_detect_transition():
    # Create a step function with noise
    series1 = np.random.normal(0, 0.1, 100)
    series2 = np.random.normal(5, 0.1, 100)
    series = np.concatenate([series1, series2])

    detector = PhaseTransitionDetector()
    transition_step = detector.detect_transition(series, window=20, threshold=5.0)

    # Should be close to 100
    assert transition_step is not None
    assert 80 <= transition_step <= 120

def test_detect_grokking_point():
    train_acc = np.concatenate([np.linspace(0, 0.95, 50), np.ones(100)])
    test_acc = np.concatenate([np.linspace(0, 0.1, 50), np.linspace(0.1, 0.95, 20), np.ones(80)])

    detector = PhaseTransitionDetector()
    step = detector.detect_grokking_point(train_acc, test_acc, train_threshold=0.9, test_threshold=0.9)

    assert step is not None
    # 50 + approx 19 steps to reach >0.9
    assert 60 <= step <= 75

def test_detect_collapse_point():
    acc = np.concatenate([np.linspace(0, 0.95, 50), np.ones(50), np.linspace(0.95, 0.2, 20), np.ones(30) * 0.2])

    detector = PhaseTransitionDetector()
    step = detector.detect_collapse_point(acc, window=5, drop_threshold=0.1)

    assert step is not None
    assert 100 <= step <= 110

def test_compute_phase_labels():
    # Construct history
    history = []

    # 0-30: learning (train low, test low)
    for i in range(30):
        history.append({"train_acc": 0.5, "test_acc": 0.1})

    # 30-60: memorization (train high, test low)
    for i in range(30, 60):
        history.append({"train_acc": 0.95, "test_acc": 0.1})

    # 60-70: transition (train high, test rising)
    for i in range(60, 70):
        history.append({"train_acc": 0.95, "test_acc": 0.5})

    # 70-100: grokking (train high, test high)
    for i in range(70, 100):
        history.append({"train_acc": 0.95, "test_acc": 0.95})

    detector = PhaseTransitionDetector()
    labels = detector.compute_phase_labels(history)

    assert len(labels) == 100
    assert labels[0] == "learning"
    assert labels[40] == "memorization"
    assert labels[65] == "transition"
    assert labels[90] == "grokking"
