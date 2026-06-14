"""
Gradient analysis tracking for grokking experiments.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def compute_gradient_noise_scale(model: nn.Module, dataloader: DataLoader, criterion=nn.CrossEntropyLoss()) -> float:
    """
    Measure gradient variance/bias ratio (simple estimator inspired by SGLD).
    We compute the gradient over the full dataset (true gradient)
    and then compute variance of mini-batch gradients.

    Args:
        model: Model
        dataloader: DataLoader containing the dataset
        criterion: Loss function
    Returns:
        Noise scale scalar
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Compute full gradient (expectation)
    model.zero_grad()
    total_loss = 0.0
    total_samples = 0

    # Accumulate full dataset gradient
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        # Weight loss by batch size to get true mean gradient
        loss = loss * (x.size(0) / len(dataloader.dataset))
        loss.backward()
        total_samples += x.size(0)

    true_grads = []
    for p in model.parameters():
        if p.grad is not None:
            true_grads.append(p.grad.clone().detach().flatten())

    if not true_grads:
        return 0.0

    true_grad_vec = torch.cat(true_grads)
    true_grad_sq_norm = torch.sum(true_grad_vec ** 2)

    # 2. Compute variance over mini-batches
    var_sum = 0.0
    model.zero_grad()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        batch_grads = []
        for p in model.parameters():
            if p.grad is not None:
                batch_grads.append(p.grad.clone().detach().flatten())

        batch_grad_vec = torch.cat(batch_grads)

        # Variance for this batch: ||g_b - g_true||^2
        var_sum += torch.sum((batch_grad_vec - true_grad_vec) ** 2).item() * x.size(0)

        model.zero_grad()

    # Average variance
    avg_var = var_sum / total_samples

    # Noise scale = variance / ||g_true||^2
    # Add epsilon to prevent division by zero
    noise_scale = avg_var / (true_grad_sq_norm.item() + 1e-10)

    return noise_scale

def track_gradient_norm(checkpoints: List[Tuple[int, nn.Module]]) -> Dict[int, float]:
    """
    Calculate gradient norms at different checkpoints.
    Note: To actually have gradients, the model needs to have backward called.
    This assumes gradients are stored in the checkpoint's parameters, or we just
    return 0.0 if not available, since we can't compute them without data here.
    If we strictly want to track historical norms, the training loop should do it.
    But we can check if gradients are attached.
    """
    history = {}
    for step, model in checkpoints:
        grad_norm_sq = 0.0
        has_grads = False
        for p in model.parameters():
            if p.grad is not None:
                has_grads = True
                grad_norm_sq += p.grad.norm().item() ** 2

        history[step] = grad_norm_sq ** 0.5 if has_grads else 0.0
    return history

def gradient_flow_analysis(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, criterion=nn.CrossEntropyLoss()) -> Dict[str, float]:
    """
    Trace gradient magnitude layer by layer.

    Args:
        model: Model
        inputs: Input batch
        targets: Target labels
        criterion: Loss function

    Returns:
        Dict mapping layer name to gradient magnitude
    """
    model.train() # Need train mode to compute gradients if there's dropout etc, though eval is safer for exact deterministic. We'll zero_grad.
    model.zero_grad()

    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    flow = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            flow[name] = param.grad.norm().item()

    return flow
