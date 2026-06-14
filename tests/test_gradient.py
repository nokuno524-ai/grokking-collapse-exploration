import torch
import torch.nn as nn
from src.gradient_analysis import compute_gradient_noise_scale, track_gradient_norm, gradient_flow_analysis
from torch.utils.data import DataLoader, TensorDataset

def test_gradient_functions():
    # Simple linear model
    model = nn.Linear(2, 2)
    criterion = nn.MSELoss()

    # Simple dataset
    X = torch.randn(10, 2)
    Y = torch.randn(10, 2)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2)

    # 1. Noise Scale
    noise_scale = compute_gradient_noise_scale(model, dataloader, criterion)
    assert isinstance(noise_scale, float)
    assert noise_scale >= 0

    # 2. Gradient flow
    flow = gradient_flow_analysis(model, X, Y, criterion)
    assert "weight" in flow
    assert "bias" in flow
    assert flow["weight"] > 0

    # 3. Track norm
    # Simulate a checkpoint list
    checkpoints = [(0, model)]
    norms = track_gradient_norm(checkpoints)
    assert 0 in norms
    assert norms[0] > 0
