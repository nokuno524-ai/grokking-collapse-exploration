import numpy as np
from src.analysis.grok_detector.stats import bootstrap_ci, permutation_test, spearman_correlation_with_ci

def test_bootstrap_ci():
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    mean, lower, upper = bootstrap_ci(data, num_bootstraps=100)
    assert lower < mean < upper

    # empty
    m, l, u = bootstrap_ci([])
    assert np.isnan(m)

def test_permutation_test():
    np.random.seed(42)
    group1 = np.random.normal(0, 1, 100)
    group2 = np.random.normal(0, 1, 100)

    p = permutation_test(group1, group2, num_permutations=100)
    # They come from same distribution, p should be relatively high
    assert p > 0.05

    group3 = np.random.normal(2, 1, 100)
    p2 = permutation_test(group1, group3, num_permutations=100)
    # Different distribution, p should be low
    assert p2 < 0.05

def test_spearman_correlation_with_ci():
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1.1, 2.1, 2.9, 4.2, 5.0])

    corr, lower, upper = spearman_correlation_with_ci(x, y, num_bootstraps=100)

    assert corr > 0.9 # Highly correlated
    assert lower <= corr <= upper
