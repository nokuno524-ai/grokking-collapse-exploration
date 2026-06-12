import pytest
import numpy as np
import torch
import math

from analysis.weight_analysis import compute_weight_norms, compute_effective_rank
from analysis.circuit_discovery import generate_importance_heatmap
from analysis.representation import compute_linear_cka, compute_representation_rank, compute_all_pairs_cka

def test_weight_norms():
    # 2x2 identity matrix
    weights = np.eye(2)
    norms = compute_weight_norms(weights)

    # L1 norm of I is 2
    assert math.isclose(norms["l1"], 2.0, rel_tol=1e-5)
    # Spectral norm (L2) of I is 1
    assert math.isclose(norms["l2"], 1.0, rel_tol=1e-5)
    # Frobenius norm of I is sqrt(2)
    assert math.isclose(norms["frobenius"], math.sqrt(2.0), rel_tol=1e-5)

    # Check with torch tensor
    tensor_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tensor_norms = compute_weight_norms(tensor_weights)
    assert math.isclose(tensor_norms["frobenius"], math.sqrt(2.0), rel_tol=1e-5)

def test_effective_rank():
    # Identity matrix should have max entropy (equal singular values)
    # Singular values are [1, 1], normalized to [0.5, 0.5]
    # Entropy = - (0.5*ln(0.5) + 0.5*ln(0.5)) = ln(2)
    # Effective rank = exp(ln(2)) = 2.0
    weights = np.eye(2)
    rank = compute_effective_rank(weights)
    assert math.isclose(rank, 2.0, rel_tol=1e-5)

    # Rank-1 matrix should have entropy 0
    # Singular values [1, 0], normalized [1, 0]
    # Entropy = 0, exp(0) = 1
    weights = np.array([[1.0, 1.0], [1.0, 1.0]])
    rank = compute_effective_rank(weights)
    assert math.isclose(rank, 1.0, rel_tol=1e-5)

def test_cka_identical():
    # CKA of identical representations should be 1.0
    np.random.seed(42)
    X = np.random.randn(10, 5)
    cka = compute_linear_cka(X, X)
    assert math.isclose(cka, 1.0, rel_tol=1e-5)

    # Test with torch tensors
    X_tensor = torch.tensor(X)
    cka_tensor = compute_linear_cka(X_tensor, X_tensor)
    assert math.isclose(cka_tensor, 1.0, rel_tol=1e-5)

def test_cka_orthogonal():
    # CKA of perfectly orthogonal representations should be 0.0
    # CKA computes correlation between similarity matrices XX^T and YY^T.
    # To make them orthogonal, we need XX^T and YY^T to have 0 dot product after centering.
    # A simple way is to use orthogonal features that don't covary.
    np.random.seed(42)
    # Create n points
    n = 10
    X = np.zeros((n, 2))
    Y = np.zeros((n, 2))

    # Make X depend only on the first 5 samples, Y on the last 5
    X[:5, 0] = np.random.randn(5)
    Y[5:, 1] = np.random.randn(5)

    # After centering, they still might have small overlap, let's just make their gram matrices orthogonal
    # Actually, linear CKA between X and Y is ||X^T Y||_F^2 / (...)
    # If X^T Y = 0 (features are orthogonal across samples), then CKA is 0.
    X = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]).T
    Y = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]).T

    cka = compute_linear_cka(X, Y)
    # the dot product of X_centered^T Y_centered might not be exactly 0,
    # let's construct explicit orthogonal centered matrices
    X_c = np.array([[1.0, -1.0, 0.0, 0.0]]).T
    Y_c = np.array([[0.0, 0.0, 1.0, -1.0]]).T

    cka = compute_linear_cka(X_c, Y_c)
    assert math.isclose(cka, 0.0, abs_tol=1e-5)

def test_representation_rank():
    np.random.seed(42)
    # Create a perfectly linear sequence, inherently rank 1
    x = np.linspace(-1, 1, 10)
    X = np.column_stack([x, 2*x, -0.5*x])

    eff_rank, entropy = compute_representation_rank(X, variance_threshold=0.99)
    assert eff_rank == 1
    assert math.isclose(entropy, 1.0, abs_tol=1e-5)

def test_all_pairs_cka():
    np.random.seed(42)
    X1 = np.random.randn(10, 5)
    X2 = X1 * 2 + 1  # Linear transform should not change CKA exactly if centered properly, but wait:
    # Actually, scaling X by c changes dot products but CKA normalizes it.
    X3 = np.random.randn(10, 5)

    reps = [X1, X2, X3]
    cka_matrix = compute_all_pairs_cka(reps)

    assert cka_matrix.shape == (3, 3)
    # Diagonal should be 1
    for i in range(3):
        assert math.isclose(cka_matrix[i, i], 1.0, rel_tol=1e-5)

    # X1 and X2 should have high similarity
    assert math.isclose(cka_matrix[0, 1], 1.0, rel_tol=1e-5)

    # Matrix should be symmetric
    assert math.isclose(cka_matrix[0, 2], cka_matrix[2, 0], rel_tol=1e-5)

def test_circuit_importance_normalization():
    from analysis.circuit_discovery import normalize_importance_scores
    # Test that normalization scales correctly
    scores = np.array([[1.0, 2.0], [3.0, 5.0]])
    norm_scores = normalize_importance_scores(scores)

    # Min should be 0, Max should be 1
    assert math.isclose(np.min(norm_scores), 0.0, rel_tol=1e-5)
    assert math.isclose(np.max(norm_scores), 1.0, rel_tol=1e-5)

    # Test element at pos (0, 1) which was 2.0
    # (2.0 - 1.0) / (5.0 - 1.0) = 1.0 / 4.0 = 0.25
    assert math.isclose(norm_scores[0, 1], 0.25, rel_tol=1e-5)

def test_circuit_importance_heatmap():
    # Test that the heatmap function runs and does not crash
    # and properly processes the normalized scores
    from analysis.circuit_discovery import normalize_importance_scores
    scores = np.random.rand(2, 4)
    scores = normalize_importance_scores(scores)

    import os
    save_path = "test_heatmap.png"
    generate_importance_heatmap(scores, save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)
