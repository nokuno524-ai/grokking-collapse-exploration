import pytest
import numpy as np
import torch
import torch.nn as nn
from src.analysis.experiment_comparison import compute_bootstrap_ci
from src.analysis.weight_analysis import compute_layerwise_statistics
from src.analysis.gradient_flow import compute_gradient_norms
from src.analysis.circuit_detection import cluster_attention_patterns

def test_compute_bootstrap_ci():
    # Test with normal distribution
    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=1.0, size=100).tolist()
    mean, lower, upper = compute_bootstrap_ci(data, n_resamples=1000)

    assert np.isclose(mean, 5.0, atol=0.2)
    assert lower < mean
    assert upper > mean

    # Test with empty list
    m, l, u = compute_bootstrap_ci([])
    assert np.isnan(m) and np.isnan(l) and np.isnan(u)

    # Test with single value
    m, l, u = compute_bootstrap_ci([3.14])
    assert m == 3.14 and l == 3.14 and u == 3.14

def test_compute_layerwise_statistics():
    model = nn.Linear(10, 5)
    nn.init.constant_(model.weight, 2.0)
    nn.init.constant_(model.bias, 1.0)

    stats = compute_layerwise_statistics(model)

    assert 'weight' in stats
    assert 'bias' in stats

    assert np.isclose(stats['weight']['mean'], 2.0)
    assert np.isclose(stats['bias']['mean'], 1.0)
    assert np.isclose(stats['weight']['std'], 0.0)

def test_compute_gradient_norms():
    model = nn.Linear(10, 5)
    # Simulate forward/backward pass
    x = torch.randn(2, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()

    grad_norms = compute_gradient_norms(model)

    assert 'weight' in grad_norms
    assert 'bias' in grad_norms
    assert grad_norms['weight'] > 0
    assert grad_norms['bias'] > 0

def test_cluster_attention_patterns():
    # Simulate attention weights: (batch=2, heads=4, seq=3, seq=3)
    attention = torch.rand(2, 4, 3, 3)
    # Normalize to sum to 1 over last dim
    attention = torch.nn.functional.softmax(attention, dim=-1)

    clusters = cluster_attention_patterns(attention, n_clusters=2)

    assert clusters.shape == (2, 4)
    assert np.max(clusters) <= 1
