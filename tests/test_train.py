import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.train import evaluate, compute_fourier_concentration
from src.model import ModularArithmeticTransformer

def test_evaluate_empty_dataloader():
    model = ModularArithmeticTransformer(prime=7)
    # Empty dataset
    dataset = TensorDataset(torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=4)
    device = torch.device("cpu")

    loss, acc = evaluate(model, dataloader, device)

    assert loss == 0.0
    assert acc == 0.0

def test_evaluate_normal():
    model = ModularArithmeticTransformer(prime=7)
    inputs = torch.randint(0, 7, (10, 2))
    targets = (inputs[:, 0] + inputs[:, 1]) % 7
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=4)
    device = torch.device("cpu")

    loss, acc = evaluate(model, dataloader, device)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_compute_fourier_concentration_normal():
    model = ModularArithmeticTransformer(prime=7)
    concentration = compute_fourier_concentration(model, top_k=2)

    assert isinstance(concentration, float)
    assert 0.0 <= concentration <= 1.0

def test_compute_fourier_concentration_zero_energy():
    model = ModularArithmeticTransformer(prime=7)
    with torch.no_grad():
        model.token_embed.weight.fill_(0.0)

    concentration = compute_fourier_concentration(model, top_k=2)
    assert concentration == 0.0
