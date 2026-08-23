import pytest
import torch
from src.data import DatasetConfig, get_task_loaders

def test_task_registry():
    config = DatasetConfig(prime=7, train_fraction=0.5, seed=42)

    tasks = ["modular_arithmetic", "polynomial_identity", "sparse_parity", "digit_sorting"]

    for task in tasks:
        train_loader, test_loader, metadata = get_task_loaders(task, config, batch_size=10)

        assert metadata["task"] == task
        assert metadata["vocab_size"] == 7
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)

        # Verify deterministic outputs (fixed seed)
        train_loader2, test_loader2, _ = get_task_loaders(task, config, batch_size=10)
        x1, y1 = next(iter(train_loader))
        x2, y2 = next(iter(train_loader2))
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)

        if task in ["sparse_parity", "digit_sorting"]:
            assert metadata["output_classes"] == 2
        else:
            assert metadata["output_classes"] == 7
