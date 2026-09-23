import pytest
import numpy as np
from src.analysis.grok_detector.stats import (
    mann_whitney_u_test,
    holm_correction,
    permutation_test,
    hedges_g,
    bootstrap_ci
)

def test_mann_whitney_u_test():
    g1 = [1, 2, 3, 4, 5]
    g2 = [10, 11, 12, 13, 14]

    stat, pval = mann_whitney_u_test(g1, g2)
    assert pval < 0.05 # Should be significantly different

def test_holm_correction():
    p_vals = [0.01, 0.04, 0.03, 0.005]
    # Sorted: 0.005 (x4), 0.01 (x3), 0.03 (x2), 0.04 (x1)
    # Expected bounds approx: 0.02, 0.03, 0.06, 0.06

    corrected = holm_correction(p_vals)

    assert np.isclose(corrected[3], 0.02)  # 0.005 * 4
    assert np.isclose(corrected[0], 0.03)  # 0.01 * 3
    assert np.isclose(corrected[2], 0.06)  # 0.03 * 2
    assert np.isclose(corrected[1], 0.06)  # max(0.04 * 1, 0.06)

def test_permutation_test():
    np.random.seed(42)
    g1 = np.random.normal(0, 1, 20).tolist()
    g2 = np.random.normal(5, 1, 20).tolist() # Very different mean

    pval = permutation_test(g1, g2, n_permutations=1000)
    assert pval < 0.05

def test_hedges_g():
    g1 = [1, 2, 3, 4, 5]
    g2 = [1, 2, 3, 4, 5]

    assert abs(hedges_g(g1, g2)) < 1e-5

    g3 = [11, 12, 13, 14, 15]
    # Large effect size
    assert hedges_g(g1, g3) < -1.0

def test_bootstrap_ci():
    np.random.seed(42)
    data = np.random.normal(10, 1, 100).tolist()

    lower, upper = bootstrap_ci(data, ci=0.95, n_resamples=1000)

    # True mean is 10, CI should likely contain it
    assert lower <= 10.5
    assert upper >= 9.5
    assert lower < upper
