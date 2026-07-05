import pytest
from analysis.grokking_detector import detect_grokking

def test_detect_grokking_clear():
    # Simulated accuracy curve that stays low then jumps
    accs = [0.1] * 20 + [0.2, 0.4, 0.7, 0.9, 0.95] + [0.98] * 20
    steps = list(range(len(accs)))

    grokked, step = detect_grokking(accs, steps, window_length=5)
    assert grokked is True
    # The jump happens around index 20-23
    assert 18 <= step <= 25

def test_detect_grokking_no_grok():
    # Curve stays low
    accs = [0.1] * 50
    steps = list(range(len(accs)))

    grokked, step = detect_grokking(accs, steps)
    assert grokked is False
    assert step is None

def test_detect_grokking_short_sequence():
    # Fallback to simple threshold
    accs = [0.1, 0.5, 0.95]
    steps = [10, 20, 30]

    # In this case it triggers the derivative logic because length is 3.
    # Because of linear spacing, the 2nd derivative argmax is at 0 or 1, which corresponds to step 10 or 20.
    grokked, step = detect_grokking(accs, steps)
    assert grokked is True
    assert step in [10, 20]

def test_detect_grokking_very_short_sequence():
    # Too short to detect
    accs = [0.1, 0.2]
    steps = [10, 20]

    grokked, step = detect_grokking(accs, steps)
    assert grokked is False
    assert step is None
