import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import ModularArithmeticTransformer
from src.train import evaluate, compute_fourier_concentration

def test_evaluate_empty_dataloader():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    dataset = TensorDataset(torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=8)
    device = torch.device('cpu')

    loss, acc = evaluate(model, dataloader, device)
    assert loss == 0.0
    assert acc == 0.0

def test_evaluate_non_empty_dataloader():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    inputs = torch.randint(0, 59, (16, 2))
    targets = torch.randint(0, 59, (16,))
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=8)
    device = torch.device('cpu')

    loss, acc = evaluate(model, dataloader, device)
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0

def test_compute_fourier_concentration():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    conc = compute_fourier_concentration(model)
    assert isinstance(conc, float)
    assert 0.0 <= conc <= 1.0
