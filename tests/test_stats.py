import numpy as np
from src.analysis.grok_detector.stats import (
    kaplan_meier_estimator,
    log_rank_test,
    permutation_test_median
)

def test_kaplan_meier_estimator():
    durations = np.array([5, 5, 8, 8, 12])
    events = np.array([1, 1, 0, 1, 1])

    times, survival, lower, upper = kaplan_meier_estimator(durations, events)

    # Check dimensions
    assert len(times) == 3
    assert len(survival) == 3
    assert len(lower) == 3
    assert len(upper) == 3

    # Check basic properties
    assert np.all(survival >= 0) and np.all(survival <= 1)
    assert np.all(lower <= survival)
    assert np.all(upper >= survival)

    # Monotonically decreasing
    assert np.all(np.diff(survival) <= 0)

def test_log_rank_test():
    d1 = np.array([5, 8, 12, 15])
    e1 = np.array([1, 1, 0, 1])

    d2 = np.array([1, 3, 5, 6])
    e2 = np.array([1, 1, 1, 1])

    chi2, p_val = log_rank_test(d1, e1, d2, e2)

    assert chi2 > 0
    assert 0 <= p_val <= 1

    # Identical groups should have non-significant p-value
    chi2_id, p_val_id = log_rank_test(d1, e1, d1, e1)
    assert p_val_id == 1.0

def test_permutation_test_median():
    d1 = np.array([10, 12, 15, 14])
    e1 = np.array([1, 1, 1, 1])

    d2 = np.array([2, 3, 4, 3])
    e2 = np.array([1, 1, 1, 1])

    p_val = permutation_test_median(d1, e1, d2, e2, n_permutations=100)

    assert 0 <= p_val <= 1

    # Identical groups should have high p-value
    p_val_id = permutation_test_median(d1, e1, d1, e1, n_permutations=100)
    assert p_val_id == 1.0
