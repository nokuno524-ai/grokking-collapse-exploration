import math
import pytest
import numpy as np
from src.rigorous_stats.cliff_stability import leave_one_out_stability, cliffs_delta

def test_cliffs_delta():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    # all y > x, so P(x > y) = 0, P(x < y) = 1. delta = -1
    assert cliffs_delta(x, y) == -1.0

    x = np.array([4, 5, 6])
    y = np.array([1, 2, 3])
    assert cliffs_delta(x, y) == 1.0

    x = np.array([1, 2])
    y = np.array([1, 2])
    assert cliffs_delta(x, y) == 0.0

def dummy_detector(steps, accs):
    return steps[0]

def test_loo_stability():
    steps_list = [[100], [200], [300]]
    accs_list = [[1], [1], [1]]

    res = leave_one_out_stability(steps_list, accs_list, dummy_detector)
    assert 'error' not in res
    assert res['all_cliffs'] == [100, 200, 300]
    assert res['median_all'] == 200.0
    # LOO medians:
    # leave 100 out: med(200, 300) = 250
    # leave 200 out: med(100, 300) = 200
    # leave 300 out: med(100, 200) = 150
    assert res['loo_medians'] == [250.0, 200.0, 150.0]
    assert res['max_deviation'] == 50.0

    res2 = leave_one_out_stability([[100]], [[1]], dummy_detector)
    assert 'error' in res2

from src.rigorous_stats.cliff_stability import effect_size_stability
from src.rigorous_stats.stats_module import cohens_d

def test_effect_size_stability():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    res = effect_size_stability(x, y, cohens_d)

    assert 'error' not in res
    assert math.isclose(res['full_estimate'], -3.0)
    assert len(res['loo_estimates']) == 6 # 3 from x, 3 from y
    assert res['max_deviation'] >= 0.0
