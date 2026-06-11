import numpy as np
from src.stats_utils import bootstrap_ci, cohens_d, bonferroni_correction, detect_phase_transition

def test_bootstrap_ci():
    # Setup reproducible data
    rng = np.random.RandomState(42)
    data = rng.normal(loc=5.0, scale=2.0, size=100)

    # Test
    stat, lower, upper = bootstrap_ci(data, statistic=np.mean, n_resamples=500, ci=0.95, seed=42)

    # Assertions
    assert isinstance(stat, float)
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower <= stat <= upper
    # True mean is 5.0, our sample should be close to that
    assert 4.0 < stat < 6.0
    assert 4.0 < lower < 6.0
    assert 4.0 < upper < 6.0

def test_cohens_d():
    # Large effect size
    g1 = [1, 2, 3, 4, 5]
    g2 = [6, 7, 8, 9, 10]

    d = cohens_d(g1, g2)
    # Mean(g1) = 3, Mean(g2) = 8
    # Var(g1) = 2.5, Var(g2) = 2.5
    # Pooled SD = sqrt(2.5) ~ 1.58
    # d = (3 - 8) / 1.58 = -3.16
    assert np.isclose(d, -3.16227766)

    # No effect size
    d = cohens_d(g1, g1)
    assert d == 0.0

def test_bonferroni_correction():
    p_values = [0.01, 0.04, 0.1, 0.001]
    adjusted, new_alpha = bonferroni_correction(p_values, alpha=0.05)

    assert new_alpha == 0.05 / 4
    # 0.01 * 4 = 0.04
    # 0.04 * 4 = 0.16
    # 0.1 * 4 = 0.4
    # 0.001 * 4 = 0.004
    assert np.allclose(adjusted, [0.04, 0.16, 0.4, 0.004])

def test_detect_phase_transition():
    # Create an obvious change point
    series = np.concatenate([np.ones(10) * 1.0, np.ones(10) * 10.0])

    # Test
    cp = detect_phase_transition(series, min_size=5, threshold=1.0)

    # It should detect the transition at index 10
    assert cp == 10

    # Test with no transition
    series_no_transition = np.ones(20)
    cp = detect_phase_transition(series_no_transition, min_size=5, threshold=1.0)
    assert cp == -1
