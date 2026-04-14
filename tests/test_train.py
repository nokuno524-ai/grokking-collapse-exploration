import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.train import compute_fourier_concentration, evaluate
from src.model import ModularArithmeticTransformer

def test_compute_fourier_concentration():
    model = ModularArithmeticTransformer(prime=59)
    conc = compute_fourier_concentration(model)
    assert isinstance(conc, float)
    assert 0.0 <= conc <= 1.0

def test_evaluate_step():
    device = torch.device('cpu')
    model = ModularArithmeticTransformer(prime=59, d_model=32)
    model.to(device)

    batch_size = 4
    inputs = torch.randint(0, 59, (batch_size, 2))
    targets = torch.randint(0, 59, (batch_size,))

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    loss, acc = evaluate(model, dataloader, device)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0
