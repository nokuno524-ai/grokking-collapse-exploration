import pytest
import numpy as np
from src.analysis.cliff import extract_cliff_stats, permutation_test, compute_ci, trend_test

def test_extract_cliff_stats_perfect():
    steps = np.linspace(0, 10000, 100)
    acc = 0.1 + (0.9 - 0.1) / (1 + np.exp(-0.01 * (steps - 5000)))
    stats = extract_cliff_stats(steps, acc)

    assert abs(stats['grokking_step'] - 5000) < 1.0
    # width = 2 * ln(9) / 0.01
    assert abs(stats['cliff_width'] - (2 * np.log(9) / 0.01)) < 1.0
    assert abs(stats['asymptotic_acc'] - 0.9) < 0.01
    assert stats['r2'] > 0.99

def test_extract_cliff_stats_plateau():
    steps = np.linspace(0, 10000, 100)
    acc = np.ones_like(steps) * 0.1
    stats = extract_cliff_stats(steps, acc)

    assert np.isnan(stats['grokking_step'])
    assert np.isnan(stats['cliff_width'])
    assert stats['asymptotic_acc'] == 0.1
    assert stats['r2'] == 0.0

def test_permutation_test_null():
    val1 = np.array([1, 2, 3, 4, 5])
    val2 = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    p = permutation_test(val1, val2, n_permutations=1000)
    assert p > 0.05 # Should not be significant

def test_permutation_test_effect():
    # Larger N so permutation p-value can be small
    val1 = np.array([1, 2, 3, 4, 5, 6, 7])
    val2 = np.array([20, 21, 22, 23, 24, 25, 26])
    p = permutation_test(val1, val2, n_permutations=1000)
    assert p < 0.05 # Should be significant

def test_permutation_test_nans():
    val1 = np.array([1, 2, np.nan, 4, 5, 6, 7])
    val2 = np.array([20, 21, 22, 23, 24, 25, np.nan])
    p = permutation_test(val1, val2, n_permutations=1000)
    assert p < 0.05

def test_compute_ci():
    val1 = np.array([1, 2, 3, 4, 5, 6, 7])
    lb, ub = compute_ci(val1)
    assert lb < np.mean(val1) < ub

    val2 = np.array([1, np.nan])
    lb, ub = compute_ci(val2)
    assert np.isnan(lb) and np.isnan(ub)

def test_trend_test():
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]
    arr3 = [7, 8, 9]
    p = trend_test([arr1, arr2, arr3])
    assert p < 0.05
