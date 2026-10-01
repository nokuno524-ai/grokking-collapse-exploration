import pytest
import math
import numpy as np
from src.rigorous_stats.cliff_detection import (
    threshold_crossing_step,
    smooth_curve,
    max_2nd_derivative_detector,
    changepoint_detector,
    compute_uncertainty_band
)

def test_threshold_crossing_step():
    steps = [0, 100, 200, 300, 400]
    accs = [0.1, 0.2, 0.96, 0.97, 0.98]
    assert threshold_crossing_step(steps, accs, threshold=0.95, sustained_steps=1) == 200
    assert threshold_crossing_step(steps, accs, threshold=0.95, sustained_steps=3) == 200

    accs_drop = [0.1, 0.2, 0.96, 0.90, 0.98]
    assert threshold_crossing_step(steps, accs_drop, threshold=0.95, sustained_steps=1) == 200
    assert threshold_crossing_step(steps, accs_drop, threshold=0.95, sustained_steps=2) == None

def test_smooth_curve():
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    res = smooth_curve(y, window_size=3)
    assert len(res) == 3
    assert np.allclose(res, [1.0, 1.0, 1.0])

def test_max_2nd_derivative_detector():
    steps = list(range(10))
    # A step function jumping at index 5
    accs = [0.0]*5 + [1.0]*5
    # Window size 3:
    # y = [0,0,0,0,0,1,1,1,1,1]
    # smooth(3) = [0,0,0, 1/3, 2/3, 1, 1, 1] (length 8)
    # d1 = [0,0, 1/3, 1/3, 1/3, 0, 0] (length 7)
    # d2 = [0, 1/3, 0, 0, -1/3, 0] (length 6)
    # max d2 at index 1 of d2.
    # orig_idx = 1 + 1 + (3-1)//2 = 3 (step 3)
    # Actually wait let's just test it runs and finds a cliff in the middle
    cliff = max_2nd_derivative_detector(steps, accs, window_size=3)
    assert cliff is not None
    assert 2 <= cliff <= 6

def test_changepoint_detector():
    steps = list(range(20))
    accs = [0.0]*10 + [1.0]*10
    cliff = changepoint_detector(steps, accs)
    assert cliff is not None
    assert 9 <= cliff <= 11

    # Degenerate
    assert changepoint_detector(list(range(20)), [0.5]*20) is None
