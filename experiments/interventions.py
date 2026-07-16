import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from copy import deepcopy

def run_weight_freezing(model: nn.Module, data_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, num_steps: int, freeze_layers: List[str] = None) -> List[float]:
    """
    Runs a training loop with specific layers frozen.

    Args:
        model: ModularArithmeticTransformer model.
        data_loader: DataLoader for the dataset.
        criterion: Loss function.
        optimizer: Optimizer.
        num_steps: Number of training steps.
        freeze_layers: List of parameter names (or substrings) to freeze.

    Returns:
        List of test accuracies at each step.
    """
    model.train()

    # Freeze specified layers
    if freeze_layers:
        for name, param in model.named_parameters():
            for freeze_name in freeze_layers:
                if freeze_name in name:
                    param.requires_grad = False

    # Initialize iterator
    iterator = iter(data_loader)
    accuracies = []

    for step in range(num_steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            x, y = next(iterator)

        # Optional: device placement could be handled here if device was passed

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        preds = out.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        accuracies.append(acc)

    return accuracies

def run_head_ablation(model: nn.Module, data_loader: torch.utils.data.DataLoader, head_idx: int) -> float:
    """
    Zeros out a specific head in the multi-head attention and measures accuracy.

    Args:
        model: ModularArithmeticTransformer model.
        data_loader: DataLoader to evaluate on.
        head_idx: Index of the head to ablate.

    Returns:
        Accuracy after ablation.
    """
    model.eval()

    # Create a copy of the model to avoid modifying the original
    ablated_model = deepcopy(model)

    layer = ablated_model.transformer.layers[0]
    out_proj = layer.self_attn.out_proj

    n_heads = ablated_model.n_heads
    d_model = ablated_model.d_model
    head_dim = d_model // n_heads

    with torch.no_grad():
        # Zero out the weights corresponding to the specified head in the output projection
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim
        out_proj.weight[:, start_idx:end_idx] = 0.0

    # Evaluate
    correct = 0
    total = 0

    # Needs device handling typically, assuming CPU here or data is on correct device
    device = next(ablated_model.parameters()).device

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = ablated_model(x)
            preds = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    if total == 0:
        return 0.0

    return correct / total

def run_finetuning_collapsed(model_collapsed: nn.Module, fresh_data_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, num_steps: int) -> List[float]:
    """
    Fine-tunes a collapsed model on fresh, pure data.

    Args:
        model_collapsed: Collapsed ModularArithmeticTransformer model.
        fresh_data_loader: DataLoader with pure data.
        criterion: Loss function.
        optimizer: Optimizer (initialized with model_collapsed's parameters).
        num_steps: Number of fine-tuning steps.

    Returns:
        List of accuracies during fine-tuning.
    """
    model_collapsed.train()

    # Unfreeze all layers just in case
    for param in model_collapsed.parameters():
        param.requires_grad = True

    iterator = iter(fresh_data_loader)
    accuracies = []

    device = next(model_collapsed.parameters()).device

    for step in range(num_steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(fresh_data_loader)
            x, y = next(iterator)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model_collapsed(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        accuracies.append(acc)

    return accuracies
