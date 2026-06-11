import pytest
from src.phase_detection import (
    detect_grokking_phase,
    detect_collapse_onset,
    compute_critical_points
)

def test_detect_grokking_phase():
    # Curve that never groks
    acc_no_grok = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.5] * 50
    assert detect_grokking_phase(acc_no_grok, threshold=0.95, window_size=10) is None

    # Curve that groks and stays up
    acc_grok = [0.1, 0.2, 0.3, 0.96, 0.97, 0.98, 0.99] + [1.0] * 20
    onset = detect_grokking_phase(acc_grok, threshold=0.95, window_size=5)
    assert onset == 3

    # Curve that temporarily spikes but doesn't stay up
    acc_spike = [0.1, 0.2, 0.98, 0.99, 0.5, 0.6] + [0.7] * 20
    assert detect_grokking_phase(acc_spike, threshold=0.95, window_size=5) is None


def test_detect_collapse_onset():
    # Baseline always high
    baseline = [1.0] * 50

    # Test curve follows baseline then degrades
    test_acc = [1.0] * 10 + [0.9] * 5 + [0.8, 0.7, 0.6, 0.5] + [0.5] * 20

    onset = detect_collapse_onset(test_acc, baseline, degradation_threshold=0.1, min_steps=5)
    # Starts degrading at index 15 (0.8), difference is 0.2 >= 0.1
    # Remains degraded for >= 5 steps
    assert onset == 15

    # Test curve never degrades significantly
    test_acc_good = [1.0] * 50
    assert detect_collapse_onset(test_acc_good, baseline, degradation_threshold=0.1, min_steps=5) is None

    # Temporary degradation (doesn't meet min_steps)
    test_acc_temp = [1.0] * 10 + [0.8, 0.8, 0.8] + [1.0] * 20
    assert detect_collapse_onset(test_acc_temp, baseline, degradation_threshold=0.1, min_steps=5) is None


def test_compute_critical_points():
    test_accuracy_dict = {
        0.0: [0.1, 0.2, 0.96, 0.98, 1.0] + [1.0] * 50,  # Groks at step 2
        0.1: [0.1, 0.2, 0.4, 0.96, 0.98] + [1.0] * 50,  # Groks at step 3
        0.2: [0.1, 0.2, 0.3, 0.4, 0.5] + [0.5] * 50,    # Fails to grok
        0.3: [0.1, 0.15, 0.2, 0.2, 0.2] + [0.2] * 50,   # Fails to grok, worse
    }

    results = compute_critical_points(test_accuracy_dict, grokking_threshold=0.95)

    assert results['collapse_threshold'] == 0.1
    assert results['grokking_onsets'][0.0] == 2
    assert results['grokking_onsets'][0.1] == 3
    assert results['grokking_onsets'][0.2] is None
    assert results['grokking_onsets'][0.3] is None

    # Recovery potential for lowest failed severity (0.2)
    # Max is 0.5, min is 0.1, diff is 0.4
    assert abs(results['recovery_potential'] - 0.4) < 1e-5
