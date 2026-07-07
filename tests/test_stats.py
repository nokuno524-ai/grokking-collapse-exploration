import numpy as np
import pytest
from analysis.statistical_tests import compute_welch_ttest, compute_ks_test, compute_bootstrap_ci, compute_cohens_d

def test_welch_ttest():
    np.random.seed(42)
    # Different means
    data1 = np.random.normal(0, 1, 100)
    data2 = np.random.normal(5, 1, 100)
    stat, p_value = compute_welch_ttest(data1, data2)
    assert p_value < 0.05

    # Same means
    data3 = np.random.normal(0, 1, 100)
    data4 = np.random.normal(0, 1, 100)
    stat2, p_value2 = compute_welch_ttest(data3, data4)
    assert p_value2 > 0.05

def test_ks_test():
    np.random.seed(42)
    # Different distributions
    data1 = np.random.normal(0, 1, 100)
    data2 = np.random.normal(5, 1, 100)
    stat, p_value = compute_ks_test(data1, data2)
    assert p_value < 0.05

    # Same distributions
    data3 = np.random.normal(0, 1, 100)
    data4 = np.random.normal(0, 1, 100)
    stat2, p_value2 = compute_ks_test(data3, data4)
    assert p_value2 > 0.05

def test_bootstrap_ci():
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)
    low, high = compute_bootstrap_ci(data)
    assert low <= np.mean(data) <= high
    assert 9 < low < 10.5
    assert 9.5 < high < 11

def test_bootstrap_ci_degenerate():
    data = [5, 5, 5, 5, 5]
    low, high = compute_bootstrap_ci(data)
    assert low == 5
    assert high == 5

def test_cohens_d():
    data1 = [1, 2, 3, 4, 5]
    data2 = [1, 2, 3, 4, 5]
    d1 = compute_cohens_d(data1, data2)
    assert np.isclose(d1, 0.0)

    data3 = [1, 2, 3]
    data4 = [4, 5, 6]
    d2 = compute_cohens_d(data3, data4)
    assert d2 < -2.0 # data3 mean is 2, data4 mean is 5
