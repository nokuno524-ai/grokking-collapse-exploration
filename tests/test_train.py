import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from src.model import ModularArithmeticTransformer
from src.train import compute_fourier_concentration, evaluate, TrainConfig, load_checkpoint
import tempfile
import os

def test_compute_fourier_concentration():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)

    conc = compute_fourier_concentration(model, top_k=3)

    assert isinstance(conc, float)
    assert 0.0 <= conc <= 1.0

def test_evaluate():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    device = torch.device("cpu")

    inputs = torch.randint(0, 11, (10, 2))
    targets = torch.randint(0, 11, (10,))
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=5)

    loss, acc = evaluate(model, dataloader, device)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss > 0
    assert 0.0 <= acc <= 1.0

def test_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")

        # Save a basic checkpoint
        state = {"model_state": {"layer.weight": torch.ones(1)}, "step": 10}
        torch.save(state, ckpt_path)

        loaded = load_checkpoint(ckpt_path)
        assert loaded["step"] == 10
        assert torch.equal(loaded["model_state"]["layer.weight"], torch.ones(1))
