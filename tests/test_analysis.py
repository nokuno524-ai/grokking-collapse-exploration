import pytest
from scripts.plot_scaling import get_onset_step

def test_get_onset_step():
    history = [
        {"step": 1, "test_acc": 0.5},
        {"step": 2, "test_acc": 0.8},
        {"step": 3, "test_acc": 0.95},
        {"step": 4, "test_acc": 0.99},
    ]

    # Test default threshold (0.9)
    assert get_onset_step(history) == 3

    # Test custom threshold
    assert get_onset_step(history, threshold=0.95) == 3
    assert get_onset_step(history, threshold=0.99) == 4

    # Test not found
    import math
    assert math.isnan(get_onset_step(history, threshold=1.0))

    # Test empty history
    assert math.isnan(get_onset_step([]))
