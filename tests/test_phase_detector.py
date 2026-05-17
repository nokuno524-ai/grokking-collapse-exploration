import pytest
import numpy as np
from src.analysis.phase_detector import (
    detect_grokking_transition,
    detect_collapse_onset,
    identify_intervention_periods
)

def test_detect_grokking_transition():
    # Gradual slow learning then spike
    # Test accuracy curve
    acc = [0.1, 0.1, 0.1, 0.15, 0.2, 0.25, 0.95, 0.98, 0.99, 1.0]
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Threshold crosses between 60 and 70. 70 is where it crosses 0.9.
    # The derivative around 70 is positive.
    transition_step = detect_grokking_transition(acc, steps, acc_threshold=0.9, window_size=3)

    assert transition_step == 70

    # Never crosses threshold
    acc_bad = [0.1] * 10
    assert detect_grokking_transition(acc_bad, steps) is None

def test_detect_collapse_onset():
    # Weight norm curve - relatively stable then explodes
    norms = [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 60.0, 120.0, 500.0]
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # Rolling average of first 5 is around 11.
    # At index 6 (step 70), norm is 60, which is > 5 * 11
    collapse_step = detect_collapse_onset(norms, steps, explosion_factor=5.0, window_size=5)

    assert collapse_step == 70

    # No collapse
    norms_stable = [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 12.0, 12.5]
    assert detect_collapse_onset(norms_stable, steps[:8], window_size=3) is None

def test_identify_intervention_periods():
    # Metric history
    metric = [1.0, 5.0, 6.0, 2.0, 1.0, 8.0, 9.0, 1.0]
    steps = [10, 20, 30, 40, 50, 60, 70, 80]

    # Periods above 4.0
    periods = identify_intervention_periods(metric, steps, threshold=4.0, condition='above')

    # Should identify two periods:
    # 1. steps[1] to steps[2] -> (20, 30)
    # 2. steps[5] to steps[6] -> (60, 70)
    assert len(periods) == 2
    assert periods[0] == (20, 30)
    assert periods[1] == (60, 70)

    # Test 'below' condition
    periods_below = identify_intervention_periods(metric, steps, threshold=3.0, condition='below')
    # steps[0] (10, 10)
    # steps[3] to steps[4] (40, 50)
    # steps[7] (80, 80)
    assert len(periods_below) == 3
    assert periods_below[0] == (10, 10)
    assert periods_below[1] == (40, 50)
    assert periods_below[2] == (80, 80)
