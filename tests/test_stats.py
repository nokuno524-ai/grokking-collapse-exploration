import numpy as np
import pytest
from src.rigorous_stats.stats_module import bootstrap_ci, permutation_test, holm_correction

def test_bootstrap_ci_basic():
    # Test with a known distribution (standard normal approx)
    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=1.0, size=1000)
    lower, upper = bootstrap_ci(data, num_bootstraps=1000, confidence_level=0.95, seed=42)

    # Expected standard error is approx 1/sqrt(1000) = 0.0316
    # 95% CI should be roughly 5.0 +/- 1.96 * 0.0316 = [4.938, 5.062]
    assert 4.9 < lower < 5.0
    assert 5.0 < upper < 5.1
    assert lower < upper

def test_bootstrap_ci_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])

def test_permutation_test_separable():
    # Two distinctly different groups
    g1 = [1, 2, 3, 1, 2]
    g2 = [10, 11, 12, 10, 11]

    p_val = permutation_test(g1, g2, num_permutations=1000, seed=42)
    assert p_val < 0.05

def test_permutation_test_inseparable():
    # Two similar groups
    g1 = [1, 2, 3, 1, 2]
    g2 = [1, 2, 3, 2, 1]

    p_val = permutation_test(g1, g2, num_permutations=1000, seed=42)
    assert p_val > 0.05

def test_permutation_test_empty():
    with pytest.raises(ValueError):
        permutation_test([], [1, 2])
    with pytest.raises(ValueError):
        permutation_test([1, 2], [])

def test_holm_correction():
    p_values = [0.01, 0.04, 0.03, 0.005]
    # Sorted: 0.005 (rank 1, mult 4), 0.01 (rank 2, mult 3), 0.03 (rank 3, mult 2), 0.04 (rank 4, mult 1)
    # Expected corrected:
    # 0.005 * 4 = 0.020
    # 0.01 * 3 = 0.030
    # 0.03 * 2 = 0.060
    # 0.04 * 1 = 0.040 -> monotonically adjusted to 0.060

    corrected = holm_correction(p_values)

    assert len(corrected) == len(p_values)
    assert np.isclose(corrected[3], 0.02)
    assert np.isclose(corrected[0], 0.03)
    assert np.isclose(corrected[2], 0.06)
    assert np.isclose(corrected[1], 0.06)

def test_holm_correction_empty():
    assert holm_correction([]) == []

def test_holm_correction_capped():
    p_values = [0.5, 0.6]
    corrected = holm_correction(p_values)
    # 0.5 * 2 = 1.0 -> 1.0
    # 0.6 * 1 = 0.6 -> capped to 1.0 because of monotonicity
    assert np.allclose(corrected, [1.0, 1.0])
