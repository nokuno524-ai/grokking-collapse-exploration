import pytest
from src.progress_measures import compute_excluded_loss, detect_phase_transition

def test_compute_excluded_loss():
    history = [
        {"step": 100, "train_loss": 0.5, "test_loss": 1.5},
        {"step": 200, "train_loss": 0.1, "test_loss": 0.5},
        {"step": 300, "train_loss": 0.01, "test_loss": 0.05},
    ]
    excluded = compute_excluded_loss(history)
    assert len(excluded) == 3
    assert excluded[0] == 1.0
    assert excluded[1] == 0.4
    assert excluded[2] == 0.04

def test_detect_phase_transition():
    history = [
        {"step": 100, "test_acc": 0.5},
        {"step": 200, "test_acc": 0.8},
        {"step": 300, "test_acc": 0.95},
        {"step": 400, "test_acc": 0.99},
    ]
    step = detect_phase_transition(history, metric="test_acc", threshold=0.9)
    assert step == 300

    step_none = detect_phase_transition(history, metric="test_acc", threshold=1.0)
    assert step_none is None
