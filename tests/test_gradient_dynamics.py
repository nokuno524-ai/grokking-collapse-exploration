import pytest
import torch
import torch.nn as nn
import numpy as np
from src.analysis.gradient_dynamics import (
    get_gradient_norms,
    estimate_gradient_noise_scale,
    calculate_gradient_coherence,
    detect_gradient_vanishing_explosion
)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def populate_dummy_grads(self, val1=1.0, val2=2.0):
        # Fake gradients
        self.fc1.weight.grad = torch.full_like(self.fc1.weight, val1)
        self.fc1.bias.grad = torch.zeros_like(self.fc1.bias)
        self.fc2.weight.grad = torch.full_like(self.fc2.weight, val2)
        self.fc2.bias.grad = torch.zeros_like(self.fc2.bias)

def test_get_gradient_norms():
    model = SimpleModel()
    model.populate_dummy_grads()

    norms = get_gradient_norms(model)

    # fc1.weight grad is 5x10 of 1.0s -> norm is sqrt(50)
    expected_fc1_norm = np.sqrt(50)
    # fc2.weight grad is 2x5 of 2.0s -> norm is sqrt(40)
    expected_fc2_norm = np.sqrt(40)

    assert 'fc1.weight' in norms
    assert 'fc2.weight' in norms
    assert 'total' in norms

    assert np.isclose(norms['fc1.weight'], expected_fc1_norm)
    assert np.isclose(norms['fc2.weight'], expected_fc2_norm)
    assert np.isclose(norms['fc1.bias'], 0.0)
    assert np.isclose(norms['total'], np.sqrt(50 + 40))

def test_estimate_gradient_noise_scale():
    # Batch 1 grads
    b1 = {
        'w1': torch.tensor([1.0, 2.0]),
        'w2': torch.tensor([0.0, 0.0])
    }
    # Batch 2 grads
    b2 = {
        'w1': torch.tensor([3.0, 4.0]),
        'w2': torch.tensor([0.0, 0.0])
    }

    noise = estimate_gradient_noise_scale([b1, b2])

    # Var for w1 element 0: var([1.0, 3.0]) = 2.0 (unbiased)
    # Var for w1 element 1: var([2.0, 4.0]) = 2.0 (unbiased)
    # Mean var = 2.0
    assert np.isclose(noise['w1'], 2.0)
    assert np.isclose(noise['w2'], 0.0)

def test_calculate_gradient_coherence():
    grad_t1 = {
        'w1': torch.tensor([1.0, 0.0]),
        'w2': torch.tensor([1.0, 1.0])
    }

    # Same direction
    grad_t2_same = {
        'w1': torch.tensor([2.0, 0.0]),
        'w2': torch.tensor([2.0, 2.0])
    }

    # Orthogonal direction
    grad_t2_ortho = {
        'w1': torch.tensor([0.0, 1.0]),
        'w2': torch.tensor([-1.0, 1.0])
    }

    coh_same = calculate_gradient_coherence(grad_t1, grad_t2_same)
    assert np.isclose(coh_same['w1'], 1.0)
    assert np.isclose(coh_same['w2'], 1.0)
    assert np.isclose(coh_same['total'], 1.0)

    coh_ortho = calculate_gradient_coherence(grad_t1, grad_t2_ortho)
    assert np.isclose(coh_ortho['w1'], 0.0)
    assert np.isclose(coh_ortho['w2'], 0.0)
    assert np.isclose(coh_ortho['total'], 0.0)

def test_detect_gradient_vanishing_explosion():
    model = SimpleModel()
    # Exploding grad
    model.fc1.weight.grad = torch.full_like(model.fc1.weight, 1000.0)
    # Vanishing grad
    model.fc2.weight.grad = torch.full_like(model.fc2.weight, 1e-8)
    # Normal grad
    model.fc1.bias.grad = torch.full_like(model.fc1.bias, 1.0)

    status = detect_gradient_vanishing_explosion(model, vanishing_threshold=1e-6, exploding_threshold=1e3)

    assert status['fc1.weight'] == 'exploding'
    assert status['fc2.weight'] == 'vanishing'
    assert status['fc1.bias'] == 'normal'
