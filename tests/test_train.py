import pytest
import torch
from src.train import TrainConfig, train, evaluate, compute_fourier_concentration
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic
from torch.utils.data import TensorDataset, DataLoader

def test_evaluate_empty_dataloader():
    model = ModularArithmeticTransformer()
    # Empty dataloader
    dataset = TensorDataset(torch.empty((0, 2), dtype=torch.long), torch.empty((0,), dtype=torch.long))
    loader = DataLoader(dataset, batch_size=32)
    device = torch.device('cpu')
    loss, acc = evaluate(model, loader, device)
    assert loss == 0.0
    assert acc == 0.0

def test_train_one_step(tmp_path):
    config = TrainConfig(max_steps=1, eval_every=1, log_every=1, save_every=1, output_dir=str(tmp_path))
    state = train(config)
    assert state.step == 1
    assert state.train_loss > 0
    assert not state.grokked

def test_compute_fourier_concentration():
    model = ModularArithmeticTransformer()
    conc = compute_fourier_concentration(model)
    assert 0.0 <= conc <= 1.0
