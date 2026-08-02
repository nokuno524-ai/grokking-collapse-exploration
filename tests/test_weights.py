import torch
import numpy as np
import pytest
from src.analysis.weights import compute_weight_statistics, compute_effective_rank

def test_compute_weight_statistics():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Standard normal distribution
    # Kurtosis should be ~0 (Fisher's), skewness ~0
    W_normal = torch.randn(1000, 1000)
    stats_normal = compute_weight_statistics(W_normal, sparsity_threshold=1e-4)

    assert np.isclose(stats_normal['kurtosis'], 0, atol=0.2)
    assert np.isclose(stats_normal['skewness'], 0, atol=0.1)
    # Sparsity for normal distribution around 0 with small threshold
    assert stats_normal['sparsity'] > 0 and stats_normal['sparsity'] < 0.01

    # Empty tensor
    W_empty = torch.tensor([])
    stats_empty = compute_weight_statistics(W_empty)
    assert stats_empty['kurtosis'] == 0.0
    assert stats_empty['skewness'] == 0.0
    assert stats_empty['sparsity'] == 0.0

def test_compute_effective_rank():
    # Rank 1 matrix: W = u @ v^T
    u = torch.randn(10, 1)
    v = torch.randn(10, 1)
    W_rank1 = u @ v.T

    # One non-zero singular value
    # Normalized s: [1, 0, ..., 0]
    # Entropy: -1 * log(1) - 0 * log(0) = 0
    # exp(0) = 1
    rank1 = compute_effective_rank(W_rank1)
    assert np.isclose(rank1, 1.0, atol=1e-5)

    # Identity matrix (scaled)
    # All singular values are equal
    # Entropy should be max = log(n)
    # exp(entropy) = n
    n = 10
    W_id = torch.eye(n)
    rank_id = compute_effective_rank(W_id)
    assert np.isclose(rank_id, n, atol=1e-5)

    # Test invalid dimensions
    with pytest.raises(ValueError):
        compute_effective_rank(torch.randn(10))
