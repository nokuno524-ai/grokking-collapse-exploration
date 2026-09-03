import numpy as np
import pytest
from src.analysis.grok_detector.detectors import piecewise_linear_detector, cusum_detector, detect_grokking_step
from src.analysis.grok_detector.stats import kaplan_meier

def test_detectors_easy_cliff():
    steps = np.arange(100) * 100
    accuracies = np.zeros(100)
    accuracies[50:] = 1.0  # Cliff at step 5000 (idx 50)

    # Piecewise
    step, band = piecewise_linear_detector(steps, accuracies)
    assert step == 4900 or step == 5000

    # CUSUM
    c_step, c_band = cusum_detector(steps, accuracies)
    # The mean is ~0.5. Residuals: first 50 are -0.5, last 50 are +0.5
    # CUSUM reaches minimum just before the jump
    assert c_step == 4900 or c_step == 5000

    # Combined
    g_step, g_band = detect_grokking_step(steps, accuracies)
    assert g_step == 4900 or g_step == 5000

def test_flat_chance_level():
    steps = np.arange(100) * 100
    accuracies = np.full(100, 1/59)
    # Add tiny noise
    accuracies += np.random.normal(0, 1e-4, 100)

    g_step, g_band = detect_grokking_step(steps, accuracies)
    assert g_step is None

def test_already_grokked():
    steps = np.arange(100) * 100
    accuracies = np.full(100, 1.0)

    g_step, g_band = detect_grokking_step(steps, accuracies)
    assert g_step == 0
    assert g_band == (0, 0)

def test_late_cliff():
    steps = np.arange(100) * 100
    accuracies = np.zeros(100)
    accuracies[-2:] = 1.0  # Cliff at the very end

    g_step, g_band = detect_grokking_step(steps, accuracies)
    # CUSUM might detect the dip slightly before the jump
    # The jump happens between index 97 (9700) and index 98 (9800)
    assert g_step in [9700, 9800, 9900]

def test_monotone_increase():
    steps = np.arange(100) * 100
    accuracies = np.linspace(0, 1, 100)

    g_step, g_band = detect_grokking_step(steps, accuracies)
    # Should find the step where it crossed grok_threshold (0.95)
    crossing_idx = np.where(accuracies >= 0.95)[0][0]
    expected_step = steps[crossing_idx]

    # Since it's a smooth curve, detectors might pick a different point,
    # but our selection rule clamps to the crossing if detectors predict too late.
    # It shouldn't be drastically off.
    assert expected_step * 0.5 <= g_step <= expected_step

def test_nans():
    steps = np.arange(10) * 100
    accuracies = np.zeros(10)
    accuracies[5:] = 1.0
    accuracies[2] = np.nan
    accuracies[8] = np.nan

    g_step, g_band = detect_grokking_step(steps, accuracies)
    # The jump is between idx 4 and 5 (steps 400 and 500)
    assert g_step in [400, 500]

def test_kaplan_meier_easy():
    # 5 seeds
    # 3 grokked at [10, 20, 30]
    # 2 censored at [50, 50]
    times = np.array([10, 20, 30, 50, 50])
    events = np.array([1, 1, 1, 0, 0])

    ut, surv = kaplan_meier(times, events)
    np.testing.assert_array_equal(ut, [10, 20, 30, 50])

    # At t=10: 5 at risk, 1 event -> surv = 4/5 = 0.8
    # At t=20: 4 at risk, 1 event -> surv = 0.8 * 3/4 = 0.6
    # At t=30: 3 at risk, 1 event -> surv = 0.6 * 2/3 = 0.4
    # At t=50: 2 at risk, 0 events, 2 censored -> surv = 0.4 * 2/2 = 0.4
    np.testing.assert_allclose(surv, [0.8, 0.6, 0.4, 0.4])
