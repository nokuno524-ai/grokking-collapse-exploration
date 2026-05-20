import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.model import ModularArithmeticTransformer
from src.train import evaluate, compute_fourier_concentration

def test_compute_fourier_concentration():
    model = ModularArithmeticTransformer(prime=7, d_model=32, n_heads=2, d_ff=64)
    # Just checking it returns a float between 0 and 1
    conc = compute_fourier_concentration(model, top_k=2)
    assert isinstance(conc, float)
    assert 0.0 <= conc <= 1.0

def test_evaluate():
    model = ModularArithmeticTransformer(prime=7, d_model=32, n_heads=2, d_ff=64)
    device = torch.device("cpu")

    # Mock data
    x = torch.randint(0, 7, (10, 2))
    y = torch.randint(0, 7, (10,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=5)

    loss, acc = evaluate(model, dataloader, device)
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss > 0
    assert 0.0 <= acc <= 1.0
