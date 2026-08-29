import pytest
import torch
import math

from scripts.weight_metrics import compute_frobenius_norm, compute_spectral_norm, compute_effective_rank

def test_frobenius_norm():
    # 2x2 matrix with elements 1, 2, 3, 4
    # Frobenius norm is sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm = compute_frobenius_norm(w)
    assert math.isclose(norm, math.sqrt(30), rel_tol=1e-5)

    # Zero matrix
    w_zero = torch.zeros((5, 5))
    assert compute_frobenius_norm(w_zero) == 0.0

def test_spectral_norm():
    # Identity matrix has singular values 1, 1... spectral norm is 1
    w = torch.eye(3)
    norm = compute_spectral_norm(w)
    assert math.isclose(norm, 1.0, rel_tol=1e-5)

    # Diagonal matrix
    w_diag = torch.diag(torch.tensor([1.0, -5.0, 2.0]))
    norm_diag = compute_spectral_norm(w_diag)
    assert math.isclose(norm_diag, 5.0, rel_tol=1e-5)

    # 1D tensor
    w_1d = torch.tensor([3.0, 4.0])
    norm_1d = compute_spectral_norm(w_1d)
    assert math.isclose(norm_1d, 5.0, rel_tol=1e-5)

def test_effective_rank():
    # Zero matrix: svd sum is 0, should return 0.0
    w_zero = torch.zeros((3, 3))
    assert compute_effective_rank(w_zero) == 0.0

    # 1D tensor: should return 1.0
    w_1d = torch.tensor([1.0, 2.0, 3.0])
    assert compute_effective_rank(w_1d) == 1.0

    # Matrix with all same singular values (e.g., identity)
    # n=3, sv=[1, 1, 1]. s_norm=[1/3, 1/3, 1/3]
    # entropy = -3 * (1/3 * log(1/3)) = -log(1/3) = log(3)
    # eff_rank = exp(log(3)) = 3.0
    w_eye = torch.eye(3)
    eff_rank = compute_effective_rank(w_eye)
    assert math.isclose(eff_rank, 3.0, rel_tol=1e-4)

    # Rank 1 matrix
    # [1, 1]
    # [1, 1]
    # Singular values: 2, 0. s_norm = [1, 0]
    # entropy = -(1*log(1) + 0*log(0)) = 0
    # eff_rank = exp(0) = 1.0
    w_rank1 = torch.ones((2, 2))
    eff_rank1 = compute_effective_rank(w_rank1)
    assert math.isclose(eff_rank1, 1.0, rel_tol=1e-4)

    # 3D tensor
    w_3d = torch.eye(2).unsqueeze(0).repeat(2, 1, 1) # shape (2, 2, 2) -> reshaped to (2, 4)
    # Values: [[1, 0, 0, 1], [1, 0, 0, 1]]
    # This is a rank 1 matrix of size 2x4. The first row is the same as the second.
    # Singular values: sqrt(2)*sqrt(2) = 2.0, 0.
    eff_rank_3d = compute_effective_rank(w_3d)
    assert math.isclose(eff_rank_3d, 1.0, rel_tol=1e-4)
