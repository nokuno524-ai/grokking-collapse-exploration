import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pytest

from src.gradient_analysis import GradientTracker
from src.loss_landscape import get_random_direction, evaluate_model_at_point, compute_1d_loss_slice
from src.fourier_tools import get_fourier_basis, compute_weight_fft, compute_fourier_concentration

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

@pytest.fixture
def model():
    return MockModel()

@pytest.fixture
def dataloader():
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=5)

def test_gradient_tracker(model, dataloader):
    tracker = GradientTracker(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        tracker.step(step_idx=1)
        optimizer.step()
        break

    assert len(tracker.steps) == 1
    assert len(tracker.grad_history['fc1.weight']) == 1
    assert len(tracker.grad_variance_history['fc1.weight']) == 1
    assert len(tracker.snr_history['fc1.weight']) == 1

    issues = tracker.check_vanishing_exploding(step_idx=1, vanishing_thresh=1e-8, exploding_thresh=1e5)
    assert isinstance(issues, dict)

def test_loss_landscape_direction(model):
    direction = get_random_direction(model)
    assert 'fc1.weight' in direction
    assert 'fc1.bias' in direction
    assert 'fc2.weight' in direction

    # Check norm scaling roughly matches
    d_norm = direction['fc1.weight'].norm().item()
    p_norm = model.fc1.weight.norm().item()
    assert np.isclose(d_norm, p_norm, rtol=1e-3)

def test_loss_landscape_1d_slice(model, dataloader):
    alphas, losses, direction = compute_1d_loss_slice(model, dataloader, torch.device('cpu'), alpha_range=(-0.1, 0.1, 3))

    assert len(alphas) == 3
    assert len(losses) == 3
    assert alphas[0] == -0.1
    assert alphas[-1] == 0.1
    assert all(loss > 0 for loss in losses)

def test_fourier_basis():
    prime = 5
    basis = get_fourier_basis(prime)
    assert basis.shape == (5, 5)
    assert basis.dtype in (torch.complex64, torch.complex128)

    # DC component should be all 1s
    assert torch.allclose(basis[0], torch.ones(5, dtype=basis.dtype))

def test_fourier_fft():
    # Construct a weight matrix with a clear periodic pattern
    prime = 7
    d_model = 4
    weights = torch.zeros((prime, d_model))

    # Add a frequency k=1 component to the first dimension
    t = torch.arange(prime, dtype=torch.float32)
    weights[:, 0] = torch.cos(2 * torch.pi * 1 * t / prime)

    spectrum = compute_weight_fft(weights, prime)

    assert spectrum.shape == (prime, d_model)
    # k=1 and k=6 (symmetric) should have high magnitude in dim 0
    assert spectrum[1, 0] > spectrum[2, 0]
    assert spectrum[6, 0] > spectrum[2, 0]

def test_fourier_concentration():
    prime = 7
    # Mock spectrum (prime, d_model)
    spectrum = torch.zeros((prime, 2))
    # DC
    spectrum[0, :] = 10.0
    # High concentration at k=1, 6
    spectrum[1, :] = 5.0
    spectrum[6, :] = 5.0
    # Low noise elsewhere
    spectrum[2:6, :] = 0.1

    concentration = compute_fourier_concentration(spectrum, top_k=2)
    # Total energy excluding DC = (5+5+0.1*4) = 10.4 per dim on avg (since same)
    # Top 2 energy = 10
    # Expected concentration ~ 10/10.4
    expected = 10.0 / 10.4
    assert np.isclose(concentration, expected, atol=1e-2)
