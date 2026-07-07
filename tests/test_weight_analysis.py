import torch
import pytest
import numpy as np
from src.weight_analysis import compute_effective_rank, compute_cosine_similarity

def test_compute_effective_rank():
    # Rank 1 matrix: all rows are identical
    w1 = torch.ones((10, 10))
    rank1 = compute_effective_rank(w1)

    # In a perfect rank-1 matrix, one singular value is non-zero, rest are zero.
    # So s_norm = [1, 0, ..., 0]
    # Shannon entropy H = -1*log(1) - 0 = 0
    # Effective rank = exp(0) = 1
    # Adding small epsilon in our implementation makes it slightly above 1
    assert rank1 < 1.1

    # Identity matrix: full rank (rank 10)
    w_full = torch.eye(10)
    rank_full = compute_effective_rank(w_full)

    # For identity matrix, all 10 singular values are 1.
    # Normalized: s_norm = [0.1, 0.1, ..., 0.1]
    # H = -10 * (0.1 * log(0.1)) = -log(0.1) = log(10)
    # Effective rank = exp(log(10)) = 10
    assert abs(rank_full - 10.0) < 0.1

def test_compute_cosine_similarity():
    # Identical matrices should have similarity 1.0
    w1 = torch.randn(5, 5)
    sim_same = compute_cosine_similarity(w1, w1)
    assert abs(sim_same - 1.0) < 1e-5

    # Orthogonal matrices should have similarity 0.0
    # Create two orthogonal vectors and reshape
    v1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    w_ortho1 = v1.view(2, 2)
    w_ortho2 = v2.view(2, 2)

    sim_ortho = compute_cosine_similarity(w_ortho1, w_ortho2)
    assert abs(sim_ortho - 0.0) < 1e-5

    # Opposite matrices should have similarity -1.0
    sim_opp = compute_cosine_similarity(w1, -w1)
    assert abs(sim_opp - (-1.0)) < 1e-5
