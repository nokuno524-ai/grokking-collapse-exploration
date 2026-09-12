import numpy as np
from src.analysis.grok_detector.detectors import fixed_threshold_detector, max_2nd_derivative_detector, plateau_detector
from src.analysis.grok_detector.stats import bootstrap_ci, benjamini_hochberg, spearman_correlation

def test_fixed_threshold_detector():
    steps = [0, 100, 200, 300, 400]
    values = [0.1, 0.4, 0.8, 0.95, 0.99]

    # Basic crossing
    assert fixed_threshold_detector(steps, values, 0.9) == 300

    # No crossing
    assert fixed_threshold_detector(steps, values, 1.0) is None

    # With smoothing (window=3)
    # 0.1, 0.4, 0.8 -> mean 0.43
    # 0.4, 0.8, 0.95 -> mean 0.71
    # 0.8, 0.95, 0.99 -> mean 0.91
    # Smoothed sequence: [0.43, 0.43, 0.43, 0.71, 0.91]
    # At step 400, smoothed is 0.91 which > 0.9
    assert fixed_threshold_detector(steps, values, 0.9, window=3) == 400

    # Edge case: empty arrays
    assert fixed_threshold_detector([], [], 0.9) is None

def test_max_2nd_derivative_detector():
    steps = [0, 100, 200, 300, 400, 500, 600]
    # Fast rise at 300
    values = [0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95]

    # d1 = [0.05, 0.05, 0.6, 0.05, 0.05, 0.05]
    # d2 = [0, 0.55, -0.55, 0, 0]
    # max d2 is at index 1 of d2. With shift +1 in code, idx=2 -> steps[2]=200
    assert max_2nd_derivative_detector(steps, values) == 200

    # No significant change
    flat_values = [0.5, 0.51, 0.52, 0.51, 0.5]
    assert max_2nd_derivative_detector(steps[:5], flat_values) is None

    # Edge case: too short
    assert max_2nd_derivative_detector([0, 100], [0.1, 0.2]) is None

def test_plateau_detector():
    steps = [0, 100, 200, 300, 400, 500, 600, 700]
    # Plateaus starting at 300
    values = [0.1, 0.4, 0.8, 0.91, 0.92, 0.89, 0.91, 0.93]

    # patience=3, threshold=0.9, noise_tol=0.05
    # Window [300, 400, 500]: [0.91, 0.92, 0.89] -> 0.89 >= (0.9 - 0.05) == 0.85. True.
    assert plateau_detector(steps, values, patience=3, plateau_threshold=0.9, noise_tol=0.05) == 300

    # Doesn't reach plateau
    low_values = [0.1] * 8
    assert plateau_detector(steps, low_values, patience=3, plateau_threshold=0.9) is None

    # Empty
    assert plateau_detector([], [], patience=3) is None

def test_bootstrap_ci():
    data = np.array([1.0, 1.0, 1.0, 1.0])
    mean, lower, upper = bootstrap_ci(data, stat_func=np.mean)
    assert mean == 1.0
    assert lower == 1.0
    assert upper == 1.0

def test_benjamini_hochberg():
    p_values = [0.01, 0.04, 0.03, 0.001, 0.5]
    rejects = benjamini_hochberg(p_values, alpha=0.05)
    # sorted p-vals: [0.001, 0.01, 0.03, 0.04, 0.5]
    # crits: [0.01, 0.02, 0.03, 0.04, 0.05]
    # Max significant is 0.04 (index 3). Thus indices 0,1,2,3 are rejected.
    # Original indices of these: 3 (0.001), 0 (0.01), 2 (0.03), 1 (0.04)
    # The last one (0.5) is not rejected.
    assert rejects == [True, True, True, True, False]

def test_spearman_correlation():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])
    corr, p = spearman_correlation(x, y)
    assert np.isclose(corr, -1.0)
    assert p < 0.05

    # Constant array
    corr_c, p_c = spearman_correlation(x, np.array([1, 1, 1, 1, 1]))
    assert np.isnan(corr_c)
    assert np.isnan(p_c)
