import pytest
import torch
import os
import shutil
from src.train import TrainConfig, train, evaluate, compute_fourier_concentration
from src.model import ModularArithmeticTransformer
from torch.utils.data import DataLoader, TensorDataset

def test_evaluate():
    model = ModularArithmeticTransformer(prime=11, d_model=32)
    device = torch.device("cpu")

    # Create fake data
    x = torch.randint(0, 11, (10, 2))
    y = torch.randint(0, 11, (10,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)

    loss, acc = evaluate(model, loader, device)

    assert loss > 0
    assert 0 <= acc <= 1.0

def test_train_short_run():
    config = TrainConfig(
        prime=11,
        d_model=32,
        n_heads=2,
        d_ff=64,
        max_steps=5,
        batch_size=32,
        output_dir="tests/temp_results",
        condition_name="test_run",
        eval_every=2,
    )

    try:
        state = train(config)
        assert state.step == 5
        assert len(state.history) > 0
        assert os.path.exists("tests/temp_results/test_run/results.json")
    finally:
        if os.path.exists("tests/temp_results"):
            shutil.rmtree("tests/temp_results")

def test_compute_fourier_concentration():
    model = ModularArithmeticTransformer(prime=11, d_model=32)
    conc = compute_fourier_concentration(model, top_k=2)
    assert 0 <= conc <= 1.0
