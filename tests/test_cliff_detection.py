"""
Unit tests for the cliff detection module.
"""

import numpy as np
import pytest

from src.rigorous_stats.cliff_detection import (
    smooth_curve,
    threshold_detector,
    changepoint_detector,
    detect_cliff
)

def test_smooth_curve():
    steps = np.arange(10)
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    smoothed = smooth_curve(steps, values, window=3)
    assert len(smoothed) == len(values)
    # the middle elements should just be the moving average
    assert np.isclose(smoothed[4], np.mean([4, 5, 6]))
    # the edges have our boundary padding
    assert smoothed[0] == 1.0

def test_threshold_detector_clean():
    steps = np.arange(0, 100, 10)
    values = np.array([0.1, 0.1, 0.1, 0.1, 0.96, 0.98, 0.99, 1.0, 1.0, 1.0])

    idx = threshold_detector(steps, values, threshold=0.95)
    assert idx == 40  # index 4

def test_threshold_detector_no_cliff():
    steps = np.arange(0, 100, 10)
    values = np.array([0.1]*10)

    idx = threshold_detector(steps, values, threshold=0.95)
    assert idx is None

def test_changepoint_detector():
    steps = np.arange(0, 60, 10)
    values = np.array([0.1, 0.1, 0.2, 0.8, 0.9, 0.9])

    # max diff is between index 2 and 3 (0.2 -> 0.8)
    idx = changepoint_detector(steps, values)
    assert idx == 30  # index 3

def test_detect_cliff_threshold():
    steps = np.arange(0, 100, 10)
    values = np.array([0.1, 0.1, 0.1, 0.1, 0.96, 0.98, 0.99, 1.0, 1.0, 1.0])

    result = detect_cliff(steps, values, method='threshold', threshold=0.95, window=3)
    assert result is not None
    # With a window of 3, the moving average of [0.1, 0.1, 0.96] is 0.386 at idx=3.
    # At idx=4, MA of [0.1, 0.96, 0.98] is 0.68.
    # At idx=5, MA of [0.96, 0.98, 0.99] is 0.976. This crosses threshold 0.95.
    # Hence the step is 50.
    assert result['cliff_step'] == 50
    assert result['pre_cliff_level'] < 0.4 # Window spans some higher values
    assert result['post_cliff_level'] > 0.9

def test_detect_cliff_changepoint():
    steps = np.arange(0, 60, 10)
    values = np.array([0.1, 0.1, 0.2, 0.8, 0.9, 0.9])

    result = detect_cliff(steps, values, method='changepoint', window=1)  # small window so it doesn't smooth away the cliff
    assert result is not None
    assert result['cliff_step'] == 30
    assert result['sharpness'] > 0

def test_detect_cliff_noisy():
    # curve that needs smoothing
    steps = np.arange(0, 100, 10)
    values = np.array([0.1, 0.9, 0.1, 0.2, 0.4, 0.9, 0.95, 0.99, 1.0, 1.0])

    result = detect_cliff(steps, values, method='threshold', threshold=0.9, window=5)
    assert result is not None
    assert result['cliff_step'] >= 40

def test_detect_cliff_double_cliff():
    steps = np.arange(0, 100, 10)
    values = np.array([0.1, 0.1, 0.5, 0.5, 0.5, 0.96, 0.98, 0.99, 1.0, 1.0])

    result = detect_cliff(steps, values, method='threshold', threshold=0.95, window=3)
    assert result is not None
    # Similar MA smoothing effect means crossing threshold of 0.95 happens around index 6 (step 60)
    assert result['cliff_step'] == 60
