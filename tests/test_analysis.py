import pytest
import numpy as np
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from analysis.statistics import bootstrap_ci, check_significance
from analysis.information import compute_entropy_binned, compute_mutual_information_binned

def test_bootstrap_ci():
    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=1.0, size=100)
    lower, upper = bootstrap_ci(data, np.mean, n_boot=100)

    assert lower < 5.0 < upper
    # Not using exact equality because bootstrap is stochastic,
    # but lower and upper should be distinct and close to mean.
    assert np.isclose(lower, 4.8, atol=0.3)
    assert np.isclose(upper, 5.2, atol=0.3)

def test_check_significance():
    np.random.seed(42)
    data1 = np.random.normal(loc=5.0, scale=1.0, size=50)
    data2 = np.random.normal(loc=5.0, scale=1.0, size=50)

    p_val = check_significance(data1, data2, n_boot=100)
    # Should not be significant
    assert p_val > 0.05

    data3 = np.random.normal(loc=10.0, scale=1.0, size=50)
    p_val2 = check_significance(data1, data3, n_boot=100)
    # Should be significant
    assert p_val2 < 0.05

def test_compute_entropy():
    # uniform distribution has high entropy
    t1 = torch.rand(1000)
    # constant has 0 entropy
    t2 = torch.ones(1000)

    e1 = compute_entropy_binned(t1, bins=10)
    e2 = compute_entropy_binned(t2, bins=10)

    assert e1 > e2
    assert np.isclose(e2, 0.0, atol=1e-5)

def test_compute_mi():
    t1 = torch.rand(1000)
    t2 = t1.clone() # Identical
    t3 = torch.rand(1000) # Independent

    mi_high = compute_mutual_information_binned(t1, t2, bins=10)
    mi_low = compute_mutual_information_binned(t1, t3, bins=10)

    assert mi_high > mi_low
    assert np.isclose(mi_low, 0.0, atol=0.1)
