import numpy as np
from src.analysis.statistics import compute_cohens_d, compute_bootstrap_ci, t_test_independent

def test_compute_cohens_d():
    g1 = np.array([1, 2, 3, 4, 5])
    g2 = np.array([2, 3, 4, 5, 6])
    d = compute_cohens_d(g1, g2)
    assert np.isclose(d, -0.63245, atol=1e-4)

def test_compute_bootstrap_ci():
    data = np.array([1, 2, 3, 4, 5])
    lower, upper = compute_bootstrap_ci(data, n_resamples=100, seed=42)
    assert lower <= 3.0 <= upper

def test_t_test_independent():
    g1 = np.array([1, 2, 3, 4, 5])
    g2 = np.array([10, 11, 12, 13, 14])
    t_stat, p_val = t_test_independent(g1, g2)
    assert p_val < 0.05
