import numpy as np
from src.analysis.grok_detector.detectors import (
    fit_sigmoid_transition,
    max_2nd_derivative_detector
)

def test_synthetic_cliff():
    steps = np.arange(0, 10000, 100)
    # create synthetic cliff at step 5000
    L, k, x0, b = 1.0, 0.02, 5000.0, 0.0

    # sigmoid formula: L / (1 + exp(-k(x-x0))) + b
    metrics = L / (1.0 + np.exp(-k * (steps - x0))) + b

    result = fit_sigmoid_transition(steps, metrics)
    assert result["success"] is True

    # transition center should be very close to 5000
    assert abs(result["center"] - 5000.0) < 100.0
    assert result["L"] > 0.9

def test_noisy_cliff():
    steps = np.arange(0, 10000, 100)
    L, k, x0, b = 1.0, 0.02, 5000.0, 0.0

    metrics = L / (1.0 + np.exp(-k * (steps - x0))) + b
    # add noise
    np.random.seed(42)
    metrics += np.random.normal(0, 0.05, len(steps))

    result = fit_sigmoid_transition(steps, metrics)
    assert result["success"] is True
    assert abs(result["center"] - 5000.0) < 200.0

    # test derivative baseline
    deriv_res = max_2nd_derivative_detector(steps, metrics, window=3)
    assert deriv_res["success"] is True
    assert abs(deriv_res["center"] - 5000.0) < 500.0

def test_no_transition():
    steps = np.arange(0, 10000, 100)
    # mostly random noise, no cliff
    np.random.seed(42)
    metrics = np.random.normal(0.1, 0.02, len(steps))

    result = fit_sigmoid_transition(steps, metrics)
    # The sigmoid fit might "succeed" but return meaningless wide bounds or very small L
    if result["success"]:
        assert result["L"] < 0.2 or result["width"] > 5000

from src.analysis.grok_detector.stats import hedges_g, permutation_test, kaplan_meier_estimator

def test_hedges_g():
    # Large effect size
    g1 = np.array([1, 2, 3, 4, 5])
    g2 = np.array([10, 11, 12, 13, 14])
    g, lower, upper = hedges_g(g1, g2)
    assert g < -2.0  # huge negative effect size
    assert upper < -1.0

    # Null effect
    g1_null = np.array([1, 2, 3, 4, 5])
    g2_null = np.array([1, 2, 3, 4, 5])
    g_null, lower_null, upper_null = hedges_g(g1_null, g2_null)
    assert abs(g_null) < 1e-5
    assert lower_null < 0 < upper_null

def test_permutation_test():
    g1 = np.array([1, 2, 3, 4, 5])
    g2 = np.array([10, 11, 12, 13, 14])
    p_val = permutation_test(g1, g2, n_permutations=1000)
    assert p_val < 0.05

    g1_null = np.array([1, 2, 3, 4, 5])
    g2_null = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    p_val_null = permutation_test(g1_null, g2_null, n_permutations=1000)
    assert p_val_null > 0.05

def test_kaplan_meier():
    durations = np.array([10, 20, 30, 40, 50])
    # 40 and 50 are right censored
    observed = np.array([True, True, True, False, False])

    km = kaplan_meier_estimator(durations, observed)

    assert len(km["timeline"]) == 6 # 0 + 5 unique times
    # At t=0, surv=1
    # At t=10, 1 event, 5 at risk -> 4/5 = 0.8
    # At t=20, 1 event, 4 at risk -> 0.8 * 3/4 = 0.6
    # At t=30, 1 event, 3 at risk -> 0.6 * 2/3 = 0.4
    # At t=40, 0 events (censored), 2 at risk -> 0.4 * 2/2 = 0.4
    # At t=50, 0 events (censored), 1 at risk -> 0.4 * 1/1 = 0.4

    assert abs(km["survival_probability"][0] - 1.0) < 1e-5
    assert abs(km["survival_probability"][1] - 0.8) < 1e-5
    assert abs(km["survival_probability"][2] - 0.6) < 1e-5
    assert abs(km["survival_probability"][3] - 0.4) < 1e-5
    assert abs(km["survival_probability"][4] - 0.4) < 1e-5
    assert abs(km["survival_probability"][5] - 0.4) < 1e-5
