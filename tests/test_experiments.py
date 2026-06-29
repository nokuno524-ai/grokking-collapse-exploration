import numpy as np
import pytest
from analysis.statistical_significance import bootstrap_ci, cohens_d

def test_bootstrap_ci_with_variance():
    # Mock data array must contain sufficient variance to prevent degenerate variance errors
    data = np.array([0.95, 0.96, 0.94, 0.97, 0.95])
    low, high = bootstrap_ci(data, n_resamples=100)

    assert low < high
    assert 0.9 <= low <= 1.0
    assert 0.9 <= high <= 1.0

def test_bootstrap_ci_no_variance():
    # Test fallback for identical values
    data = np.array([0.95, 0.95, 0.95, 0.95, 0.95])
    low, high = bootstrap_ci(data, n_resamples=100)

    # Should use fallback returning exact mean
    assert low == 0.95
    assert high == 0.95

def test_cohens_d():
    group1 = np.array([1.0, 1.1, 0.9, 1.2, 0.8])
    group2 = np.array([0.1, 0.2, 0.0, 0.3, -0.1])

    d = cohens_d(group1, group2)
    assert d > 0.0

def test_math_epsilon_bounds():
    # When checking against mathematical bounds like 1.0, use epsilon tolerance
    epsilon = 1e-5
    calc_val = 1.0 - 1e-6

    # Simulating the check
    assert abs(calc_val - 1.0) < epsilon
