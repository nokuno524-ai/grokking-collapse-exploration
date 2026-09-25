import math
import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rigorous_stats.cliff_detection import (
    threshold_crossing_step,
    smooth_curve,
    max_2nd_derivative_detector,
    changepoint_detector,
    compute_uncertainty_band
)

from rigorous_stats.stats_module import (
    cohens_d,
    wilson_score_interval,
    compare_cliff_distributions,
    compute_grokking_rates
)


def create_synthetic_curve(type='sharp_cliff', n_steps=100, cliff_start=50):
    steps = list(range(n_steps))
    accs = np.zeros(n_steps)

    if type == 'sharp_cliff':
        # 0.0 -> 1.0 instantly
        accs[cliff_start:] = 1.0
    elif type == 'smooth_cliff':
        # Sigmoid
        accs = 1.0 / (1.0 + np.exp(-0.5 * (np.array(steps) - cliff_start)))
    elif type == 'double_step':
        # Step to 0.5, then step to 1.0
        accs[cliff_start//2:cliff_start] = 0.5
        accs[cliff_start:] = 1.0
    elif type == 'no_cliff':
        accs[:] = 0.1

    return steps, accs.tolist()


def test_synthetic_sharp_cliff():
    steps, accs = create_synthetic_curve('sharp_cliff', n_steps=100, cliff_start=50)

    # max_2nd_deriv
    assert max_2nd_derivative_detector(steps, accs, window_size=3) in [48, 49, 50, 51]

    # changepoint
    cp = changepoint_detector(steps, accs)
    assert cp is not None
    assert 49 <= cp <= 51

    # threshold
    assert threshold_crossing_step(steps, accs, threshold=0.95) == 50


def test_synthetic_smooth_cliff():
    steps, accs = create_synthetic_curve('smooth_cliff', n_steps=100, cliff_start=50)

    # max_2nd_deriv will detect the start of the acceleration phase, which is before the midpoint (50)
    # the exact location depends on window size, but it should be < 50
    m2 = max_2nd_derivative_detector(steps, accs, window_size=5)
    assert m2 is not None and m2 < 50

    # changepoint should nail the midpoint
    cp = changepoint_detector(steps, accs)
    assert cp is not None
    assert 48 <= cp <= 52


def test_synthetic_no_cliff():
    steps, accs = create_synthetic_curve('no_cliff', n_steps=100)

    assert max_2nd_derivative_detector(steps, accs) is None
    assert changepoint_detector(steps, accs) is None
    assert threshold_crossing_step(steps, accs) is None

    med, lb, ub = compute_uncertainty_band(steps, accs)
    assert med is None and lb is None and ub is None


def test_uncertainty_band():
    steps, accs = create_synthetic_curve('sharp_cliff', n_steps=100, cliff_start=50)

    med, lb, ub = compute_uncertainty_band(steps, accs, method='max_2nd_derivative')
    assert med is not None and lb is not None and ub is not None
    assert lb <= med <= ub
    assert 45 <= med <= 55


def test_edge_cases():
    assert threshold_crossing_step([], []) is None

    with pytest.raises(ValueError):
        threshold_crossing_step([1, 2], [1.0])

    steps, accs = create_synthetic_curve('sharp_cliff', n_steps=100, cliff_start=50)
    accs[60] = float('nan')
    accs[61] = None

    # Should handle NaNs gracefully without crashing
    res = max_2nd_derivative_detector(steps, accs)
    assert res is not None


def test_stats_module_correctness():
    # Cohen's d
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 4, 5, 6, 7])
    d = cohens_d(x, y)
    assert d < 0  # x is smaller than y

    # Binomial CI
    lb, ub = wilson_score_interval(10, 10) # 100% success
    assert lb > 0.6 # should not be just [1, 1], lower bound should be reasonably lower than 1
    assert math.isclose(ub, 1.0, rel_tol=1e-5)

    # Cliff distributions
    x_steps = [100, 105, 95]
    y_steps = [200, 205, 195]
    res = compare_cliff_distributions(x_steps, y_steps)
    assert res['n1'] == 3
    assert res['mean1'] == 100.0
    assert res['mean2'] == 200.0
    assert res['mwu_pvalue'] <= 0.1 # strong difference with n=3 is exact p=0.1

    # Grokking rates
    res2 = compute_grokking_rates([True, True, False], [False, False, False])
    assert math.isclose(res2['rate1'], 2/3)
    assert res2['rate2'] == 0.0
