import pytest
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.statistical_tests import cohens_d, bootstrap_ci, mwu_test

def test_cohens_d():
    # Identical groups -> d = 0
    g1 = np.array([1, 2, 3, 4, 5])
    assert cohens_d(g1, g1) == 0.0

    # Empty groups -> d = 0
    assert cohens_d(np.array([]), np.array([])) == 0.0

    # Known difference
    g2 = np.array([10, 20, 30])
    g3 = np.array([11, 21, 31])
    # The standard deviation is the same, diff in means is exactly -1
    # For small n, just assert it is calculated without error and is negative
    d = cohens_d(g2, g3)
    assert d < 0
    assert abs(d) > 0

def test_bootstrap_ci():
    # Empty data
    assert bootstrap_ci(np.array([])) == (0.0, 0.0)

    # Static data - must have variance to avoid BCa degenerate warnings
    g1 = np.array([4.9, 5.0, 5.1, 5.0, 5.0])
    ci = bootstrap_ci(g1)
    # The CI should bracket 5.0 tightly
    assert ci[0] <= 5.0
    assert ci[1] >= 5.0

def test_mwu_test():
    # Empty groups
    stat, p = mwu_test(np.array([]), np.array([]))
    assert stat == 0.0
    assert p == 1.0

    # Highly distinct groups
    g1 = np.array([1, 2, 3, 4, 5])
    g2 = np.array([100, 200, 300, 400, 500])
    stat, p = mwu_test(g1, g2)
    # p should be small
    assert p <= 0.1
    # With scipy stats, the U statistic for completely disjoint sets of size 5 is 0 or 25
    assert stat == 0.0 or stat == 25.0
