import torch
import pytest
from src.model import ModularArithmeticTransformer
from analysis.weight_space import compute_weight_metrics, measure_weight_sparsity, correlate_metrics, compute_hessian_eigenvalues

def test_compute_weight_metrics():
    model = ModularArithmeticTransformer()
    metrics = compute_weight_metrics(model)

    assert "weight_norm" in metrics
    assert "spectral_norm" in metrics
    assert "effective_rank" in metrics

    assert metrics["weight_norm"] > 0
    assert metrics["spectral_norm"] > 0
    assert metrics["effective_rank"] > 0

def test_measure_weight_sparsity():
    model = ModularArithmeticTransformer()
    # High threshold should include all weights
    sparsity = measure_weight_sparsity(model, threshold=100.0)
    assert sparsity == 1.0

    # Sparsity with tiny threshold should be ~0.0
    sparsity = measure_weight_sparsity(model, threshold=1e-10)
    assert sparsity >= 0.0 and sparsity < 0.1

def test_correlate_metrics():
    # Perfectly correlated
    m1 = [1.0, 2.0, 3.0]
    m2 = [2.0, 4.0, 6.0]
    corr = correlate_metrics(m1, m2)
    assert abs(corr - 1.0) < 1e-5

    # Perfectly anti-correlated
    m3 = [-1.0, -2.0, -3.0]
    corr_anti = correlate_metrics(m1, m3)
    assert abs(corr_anti - (-1.0)) < 1e-5

def test_compute_hessian_eigenvalues():
    # Create a small non-transformer model for testing to avoid
    # PyTorch FlashAttention backward issues on CPU during HVP
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 59)

        def forward(self, x):
            return self.linear(x.float())

    model = SimpleModel()

    # Mock dataloader
    x = torch.randint(0, 59, (4, 2))
    y = torch.randint(0, 59, (4,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    criterion = torch.nn.CrossEntropyLoss()

    # Use low iterations for speed in testing
    eigenvalues = compute_hessian_eigenvalues(model, loader, criterion, top_k=2, num_iters=2)

    assert len(eigenvalues) == 2
