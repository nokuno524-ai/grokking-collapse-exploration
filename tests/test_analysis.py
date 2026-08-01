import torch
import numpy as np
import pytest
from src.analysis.weights import get_layer_norms, get_weight_distributions, get_effective_ranks
from src.analysis.information import compute_cka, compute_mutual_information
from src.analysis.phase_transition import detect_grokking_transition, detect_collapse_onset
from src.analysis.dynamics import track_gradient_norms, compute_gradient_noise_scale, estimate_hessian_eigenvalues
from src.model import ModularArithmeticTransformer

def test_effective_rank():
    # Construct a rank-1 matrix: an outer product of two vectors
    u = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(1)
    v = torch.tensor([1.0, 1.0, 1.0]).unsqueeze(0)
    w = u @ v # shape (3,3)

    # We can inject this into a dummy model or mock it
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3, bias=False)
            self.linear.weight.data = w

    model = DummyModel()
    ranks = get_effective_ranks(model)

    # Since it's exactly rank 1, singular values are [sqrt(3), 0, 0].
    # Normalizing it gives [1, 0, 0]. Entropy is 0. Exp(0) = 1.
    assert np.isclose(ranks['linear.weight'], 1.0, atol=1e-5)

def test_cka():
    # Identical features should yield CKA of 1.0
    x = torch.randn(10, 5)
    cka = compute_cka(x, x)
    assert np.isclose(cka, 1.0, atol=1e-5)

    # Orthogonal features should yield very low CKA
    # We can make orthogonal matrices
    x = torch.eye(5)
    y = torch.zeros(5, 5) # Not orthogonal per se, but CKA with zeros will be 0
    # Wait, gram of zeros is 0, norm is 0, so CKA handles this by adding 1e-10, yielding ~0.
    cka = compute_cka(x, y)
    assert np.isclose(cka, 0.0, atol=1e-5)

def test_phase_transition():
    # Test grokking transition
    # Accuracy stays low, then jumps to 0.95 at index 3
    accs = np.array([0.1, 0.2, 0.8, 0.95, 0.96])
    assert detect_grokking_transition(accs, threshold=0.9) == 3

    # No grokking
    accs = np.array([0.1, 0.2, 0.3])
    assert detect_grokking_transition(accs, threshold=0.9) == -1

    # Test collapse onset (acceleration of negative slope)
    norms = np.array([1.0, 0.9, 0.8, 0.4, 0.1, 0.05])
    # diff1 = [-0.1, -0.1, -0.4, -0.3, -0.05]
    # diff2 = [0, -0.3, 0.1, 0.25]
    # argmin(diff2) is index 1.
    # The returned onset should be 1 + 1 = 2
    assert detect_collapse_onset(norms) == 2

def test_hessian_eigenvalues():
    # Use a small synthetic model to ensure the eigenvalue doesn't crash
    model = ModularArithmeticTransformer()
    # Mock dataloader
    dataset = torch.utils.data.TensorDataset(torch.randint(0, 59, (4, 2)), torch.randint(0, 59, (4,)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    criterion = torch.nn.CrossEntropyLoss()

    eig = estimate_hessian_eigenvalues(model, dataloader, criterion, torch.device('cpu'), max_iters=2)
    assert isinstance(eig, float)
