import pytest
import numpy as np
import math
from src.rigorous_stats.stats_module import cohens_d, compare_cliff_distributions

def test_cohens_d():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    # pool_var = (2*1 + 2*1)/4 = 1 -> pool_std = 1
    # mean_x = 2, mean_y = 5 -> d = -3 / 1 = -3.0
    assert math.isclose(cohens_d(x, y), -3.0)

    # NaN cases
    assert math.isnan(cohens_d(np.array([1]), np.array([1, 2])))
    assert cohens_d(np.array([1, 1]), np.array([1, 1])) == 0.0

def test_compare_cliff_distributions():
    x = [100, 200, 300]
    y = [400, 500, 600]
    res = compare_cliff_distributions(x, y)
    assert res['n1'] == 3
    assert res['n2'] == 3
    assert res['mean1'] == 200.0
    assert res['mean2'] == 500.0
    assert res['median1'] == 200.0
    assert math.isclose(res['cohens_d'], -3.0)
    assert 'mwu_pvalue' in res

from src.rigorous_stats.stats_module import estimate_slopes

def test_estimate_slopes():
    steps = [0, 10, 20, 30, 40, 50]
    # Before cliff (step 20), acc is 0, 0.1, 0.2 (slope 0.01)
    # After cliff (step 20), acc is 0.2, 0.8, 0.9, 1.0 (slope ...)
    accs = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]
    res = estimate_slopes(steps, accs, cliff_step=20)
    assert math.isclose(res['slope_before'], 0.01)
    assert 'slope_after' in res

    # Degenerate cases
    res2 = estimate_slopes([0, 10], [0.0, 1.0], cliff_step=100)
    assert math.isnan(res2['slope_before'])

from src.rigorous_stats.stats_module import bootstrap_ci_effect_size, bootstrap_ci_slope, cohens_d

def test_bootstrap_ci_effect_size():
    # Set seed for reproducibility
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 10])

    ci = bootstrap_ci_effect_size(x, y, cohens_d, n_bootstraps=100)
    assert ci[0] < ci[1]
    assert not math.isnan(ci[0])

def test_bootstrap_ci_slope():
    np.random.seed(42)
    steps = [0, 10, 20, 30, 40, 50]
    accs = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]
    cis = bootstrap_ci_slope(steps, accs, cliff_step=20, n_bootstraps=100)

    assert cis['slope_before_ci'][0] <= cis['slope_before_ci'][1]
    assert cis['slope_after_ci'][0] <= cis['slope_after_ci'][1]
