import pytest
import torch
import torch.nn as nn
from analysis.weight_analysis import WeightAnalysisSuite

def test_effective_rank():
    model = nn.Linear(10, 10)
    suite = WeightAnalysisSuite(model)

    # Create a rank 1 matrix
    u = torch.randn(10, 1)
    v = torch.randn(1, 10)
    rank1_matrix = u @ v

    # Create a full rank identity matrix
    identity_matrix = torch.eye(10)

    rank1_er = suite.get_effective_rank(rank1_matrix)
    identity_er = suite.get_effective_rank(identity_matrix)

    # Effective rank of rank 1 matrix should be close to 1
    assert rank1_er == pytest.approx(1.0, rel=1e-2)
    # Effective rank of identity matrix of size 10x10 is 10
    assert identity_er == pytest.approx(10.0, rel=1e-2)

def test_weight_norm():
    model = nn.Linear(3, 4, bias=False)
    # Set weights to ones. Matrix has 12 elements. L2 norm should be sqrt(12)
    nn.init.ones_(model.weight)

    suite = WeightAnalysisSuite(model)
    norm = suite.get_weight_norm()

    assert norm == pytest.approx((12.0) ** 0.5, rel=1e-4)

def test_weight_connectivity():
    class TwoLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(5, 5, bias=False)
            self.l2 = nn.Linear(5, 5, bias=False)

    model = TwoLayer()
    # Set both to ones, they are identical
    nn.init.ones_(model.l1.weight)
    nn.init.ones_(model.l2.weight)

    suite = WeightAnalysisSuite(model)
    cos_sim = suite.get_weight_connectivity('l1.weight', 'l2.weight')

    # Identical weights -> cosine similarity of 1.0
    assert cos_sim == pytest.approx(1.0, rel=1e-4)

    # Change l2 to be orthogonal/different
    nn.init.zeros_(model.l2.weight)
    model.l2.weight.data[0, 0] = 1.0
    # cos sim between [1,1,1...] and [1,0,0...] = 1 / sqrt(25) = 0.2
    cos_sim_diff = suite.get_weight_connectivity('l1.weight', 'l2.weight')
    assert cos_sim_diff == pytest.approx(0.2, rel=1e-4)

def test_hessian_eigenvalues():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([2.0, -1.0]))

        def forward(self, x):
            return x * self.w

    model = SimpleModel()
    suite = WeightAnalysisSuite(model)

    inputs = torch.tensor([[1.0, 1.0]])
    targets = torch.tensor([[2.0, 2.0]])
    loss_fn = nn.MSELoss()

    # We just want to ensure it runs without crashing and returns a float list
    eigenvals = suite.estimate_hessian_eigenvalues(loss_fn, inputs, targets, k=1, num_iters=5)

    assert len(eigenvals) == 1
    assert isinstance(eigenvals[0], float)

def test_gradient_flow():
    model = nn.Linear(2, 2, bias=False)
    suite = WeightAnalysisSuite(model)

    # Set gradients manually
    model.weight.grad = torch.ones(2, 2) * 2.0

    flow = suite.analyze_gradient_flow()

    assert 'weight' in flow
    # Norm of a 2x2 matrix of 2s is sqrt(16) = 4
    assert flow['weight'] == pytest.approx(4.0, rel=1e-4)
