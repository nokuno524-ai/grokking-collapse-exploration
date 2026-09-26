import torch
from src.analysis.representational_geometry import compute_cka, compute_effective_rank, compute_probe_accuracy

def test_cka_identical():
    X = torch.randn(10, 5)
    assert abs(compute_cka(X, X) - 1.0) < 1e-5

def test_cka_orthogonal():
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    Y = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    # Compute dot products manual verification could be added
    print("CKA orthogonal:", compute_cka(X, Y))

def test_effective_rank():
    # Identity matrix should have rank D
    D = 5
    X = torch.eye(D)
    eps = 1e-10
    # True rank is exactly 5
    r = compute_effective_rank(X, eps)
    assert abs(r - D) < 1e-4

    # Rank 1 matrix
    X = torch.ones(10, D)
    r = compute_effective_rank(X, eps)
    # The effective rank should be close to 1
    assert abs(r - 1.0) < 1e-4

def test_probe_accuracy():
    # Synthetic separable data
    N = 100
    D = 2
    # Two clusters
    X = torch.cat([torch.randn(N//2, D) + torch.tensor([5.0, 5.0]),
                   torch.randn(N//2, D) + torch.tensor([-5.0, -5.0])])
    y = torch.cat([torch.zeros(N//2, dtype=torch.long),
                   torch.ones(N//2, dtype=torch.long)])

    acc = compute_probe_accuracy(X, y, lr=0.1, epochs=50)
    assert acc > 0.95

def test_cka_error_paths():
    X = torch.randn(10, 5)
    Y = torch.randn(5, 5) # Mismatched shapes in N
    try:
        compute_cka(X, Y)
        assert False, "Should have thrown an exception on N shape mismatch"
    except Exception:
        pass

    X = torch.zeros(10, 5)
    Y = torch.zeros(10, 5)
    cka = compute_cka(X, Y)
    assert cka == 0.0
