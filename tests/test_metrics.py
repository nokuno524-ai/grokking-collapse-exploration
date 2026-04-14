import pytest
from src.progress_measures import (
    compute_excluded_loss,
    compute_learning_speed,
    detect_phase_transition,
    analyze_grokking_trajectory
)

def test_compute_excluded_loss():
    history = [
        {"train_loss": 0.1, "test_loss": 1.0},
        {"train_loss": 0.05, "test_loss": 0.8}
    ]
    excluded = compute_excluded_loss(history)
    assert excluded == [pytest.approx(0.9), pytest.approx(0.75)]

def test_compute_learning_speed():
    history = [
        {"step": 0, "test_acc": 0.1},
        {"step": 100, "test_acc": 0.2},
        {"step": 200, "test_acc": 0.5},  # Jump!
    ]
    # window size 2 -> difference is steps i - (i-window)
    speeds = compute_learning_speed(history, window=2)
    assert speeds[0]["test_acc_speed"] == 0.0
    assert speeds[1]["test_acc_speed"] == 0.0
    # at index 2, history[2] vs history[0] -> test_acc: 0.5 - 0.1 = 0.4. steps: 200 - 0 = 200. speed = 0.4 / 200 * 1000 = 2.0
    assert speeds[2]["test_acc_speed"] == pytest.approx(2.0)

def test_detect_phase_transition():
    history = [
        {"step": 10, "test_acc": 0.5},
        {"step": 20, "test_acc": 0.8},
        {"step": 30, "test_acc": 0.95},
        {"step": 40, "test_acc": 0.99},
    ]
    transition = detect_phase_transition(history, metric="test_acc", threshold=0.9)
    assert transition == 30

    transition_high = detect_phase_transition(history, metric="test_acc", threshold=1.0)
    assert transition_high is None

def test_analyze_grokking_trajectory():
    history = [
        {"step": 10, "train_acc": 0.5, "test_acc": 0.1, "fourier_concentration": 0.0, "weight_norm": 10.0},
        {"step": 20, "train_acc": 0.995, "test_acc": 0.1, "fourier_concentration": 0.05, "weight_norm": 12.0},  # Memorization complete
        {"step": 30, "train_acc": 1.0, "test_acc": 0.2, "fourier_concentration": 0.2, "weight_norm": 15.0},   # Circuit formation onset (>0.1 and > prev*1.5)
        {"step": 40, "train_acc": 1.0, "test_acc": 0.96, "fourier_concentration": 0.5, "weight_norm": 5.0},   # Grokking step (>0.95)
    ]
    analysis = analyze_grokking_trajectory(history)

    assert analysis["phases_detected"] is True
    assert analysis["memorization_complete_step"] == 20
    assert analysis["circuit_formation_onset"] == 30
    assert analysis["grokking_step"] == 40
    assert analysis["delay_mem_to_grok"] == 20
    assert analysis["max_weight_norm"] == 15.0
    assert analysis["min_weight_norm"] == 5.0
    assert analysis["weight_norm_reduction"] == 10.0
