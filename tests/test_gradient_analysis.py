import torch
import torch.nn as nn
from src.analysis.gradient_analysis import compute_gradient_noise_scale, compute_gradient_coherence, GradientTracker

def test_compute_gradient_noise_scale():
    batch_size = 32

    # If batch grad is exactly full grad, noise scale should be 0
    full_grad = torch.randn(100)
    batch_grad = full_grad.clone()

    scale = compute_gradient_noise_scale(batch_grad, full_grad, batch_size)
    assert scale == 0.0

    # Let batch_grad = full_grad + noise
    noise = torch.randn(100) * 0.1
    batch_grad = full_grad + noise

    scale = compute_gradient_noise_scale(batch_grad, full_grad, batch_size)
    assert scale > 0.0

def test_compute_gradient_coherence():
    # If gradients are perfectly aligned, coherence should be 1.0
    grads = [torch.ones(10), torch.ones(10), torch.ones(10)]
    coherence = compute_gradient_coherence(grads)
    assert abs(coherence - 1.0) < 1e-5

    # If orthogonal, coherence should be 0
    grads = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    coherence = compute_gradient_coherence(grads)
    assert abs(coherence - 0.0) < 1e-5

    # If opposite, coherence should be -1
    grads = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    coherence = compute_gradient_coherence(grads)
    assert abs(coherence - (-1.0)) < 1e-5

def test_gradient_tracker():
    model = nn.Linear(10, 2)
    tracker = GradientTracker(model)

    # Fake gradients
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()

    tracker.log_gradient_norm()
    norms = tracker.get_norms()

    assert len(norms) == 1
    assert norms[0] > 0.0
