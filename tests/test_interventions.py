import torch
import pytest
from src.model import ModularArithmeticTransformer
from experiments.interventions import run_weight_freezing, run_head_ablation, run_finetuning_collapsed

def test_run_weight_freezing():
    model = ModularArithmeticTransformer(d_model=16, n_heads=1, d_ff=32)
    x = torch.randint(0, 59, (4, 2))
    y = torch.randint(0, 59, (4,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Freeze token_embed
    accuracies = run_weight_freezing(model, loader, criterion, optimizer, num_steps=2, freeze_layers=["token_embed"])

    assert len(accuracies) == 2
    assert not model.token_embed.weight.requires_grad
    # Just to be sure, check another layer requires grad
    assert model.output_head.weight.requires_grad

def test_run_head_ablation():
    model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32)
    x = torch.randint(0, 59, (4, 2))
    y = torch.randint(0, 59, (4,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    acc = run_head_ablation(model, loader, head_idx=0)

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_run_finetuning_collapsed():
    model = ModularArithmeticTransformer(d_model=16, n_heads=1, d_ff=32)

    # Simulate a collapsed model by freezing it entirely
    for param in model.parameters():
        param.requires_grad = False

    x = torch.randint(0, 59, (4, 2))
    y = torch.randint(0, 59, (4,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    accuracies = run_finetuning_collapsed(model, loader, criterion, optimizer, num_steps=2)

    assert len(accuracies) == 2
    # Ensure it was unfrozen
    for param in model.parameters():
        assert param.requires_grad
