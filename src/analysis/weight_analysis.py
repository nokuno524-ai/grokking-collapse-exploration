import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import copy

def get_weight_norms(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Track per-layer weight norms (L1, L2, spectral, Frobenius).
    """
    norms = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            w = param.detach()
            if w.dim() < 2:
                w_flat = w.view(-1)
                l1 = torch.norm(w_flat, p=1).item()
                l2 = torch.norm(w_flat, p=2).item()
                spectral = l2 # for 1D
                frob = l2
            else:
                w_2d = w.reshape(w.size(0), -1)
                l1 = torch.norm(w_2d, p=1).item()
                l2 = torch.norm(w_2d, p=2).item()
                # Use SVD for spectral norm
                _, S, _ = torch.linalg.svd(w_2d)
                spectral = S[0].item() if len(S) > 0 else 0.0
                frob = torch.norm(w_2d, p='fro').item()

            norms[name] = {
                'l1': l1,
                'l2': l2,
                'spectral': spectral,
                'frobenius': frob
            }
    return norms

def compute_effective_rank(model: nn.Module) -> Dict[str, float]:
    """
    Compute weight effective rank evolution via singular value entropy of weight matrices.
    """
    ranks = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.detach()
            w_2d = w.reshape(w.size(0), -1)
            _, S, _ = torch.linalg.svd(w_2d)
            if len(S) > 0:
                s_norm = S / S.sum()
                # Entropy
                entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
                eff_rank = torch.exp(entropy).item()
                ranks[name] = eff_rank
    return ranks

def measure_weight_sparsity(model: nn.Module, threshold: float = 1e-3) -> Dict[str, float]:
    """
    Measure weight sparsity (fraction of near-zero weights, threshold-based) over training.
    """
    sparsity = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            w = param.detach()
            frac_near_zero = (w.abs() < threshold).float().mean().item()
            sparsity[name] = frac_near_zero
    return sparsity

def track_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Track gradient norms per layer. Assumes loss.backward() has been called.
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.detach()
            grad_norms[name] = torch.norm(g, p=2).item()
    return grad_norms

def compute_hessian_max_eigenvalue(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    num_iterations: int = 10,
    device: str = 'cpu'
) -> float:
    """
    Compute Hessian eigenvalue approximations (via power iteration) to detect loss landscape sharpness.
    Requires device='cpu' and flash attention disabled as per memory constraints when using cpu backward.
    """
    import torch.nn.functional as F

    # Ensure flash attention is disabled for CPU backward passes
    torch.backends.native_sdp_enable_flash = False
    torch.backends.native_sdp_enable_math = True

    model.eval()

    # Get a batch
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0].to(device), batch[1].to(device)
    else:
        inputs = batch.to(device)
        targets = None

    outputs = model(inputs)
    if targets is not None:
        loss = criterion(outputs, targets)
    else:
        loss = criterion(outputs)

    # Get gradients
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # Initialize random vector v
    v = [torch.randn_like(p).to(device) for p in model.parameters()]
    # Normalize v
    norm = torch.sqrt(sum(torch.sum(x ** 2) for x in v))
    v = [x / norm for x in v]

    eigenvalue = 0.0
    for _ in range(num_iterations):
        # Compute Hv
        Hv = torch.autograd.grad(grads, model.parameters(), grad_outputs=v, retain_graph=True)

        # Compute Rayleigh quotient: v^T H v
        eigenvalue = sum(torch.sum(h * vi) for h, vi in zip(Hv, v)).item()

        # Update v
        norm = torch.sqrt(sum(torch.sum(h ** 2) for h in Hv))
        v = [h / norm for h in Hv]

    return eigenvalue

def plot_metrics_across_collapse_levels(
    results_dict: Dict[str, Dict[str, List[float]]],
    metric_name: str,
    save_path: str = None
):
    """
    Compare ALL metrics across collapse levels (pure, low, medium, severe, high) in unified plots.
    results_dict format: { collapse_level: { 'steps': [1,2,...], 'metric_values': [0.1, 0.2, ...] } }
    """
    plt.figure(figsize=(10, 6))

    colors = {
        'pure': 'blue',
        'low': 'green',
        'medium': 'orange',
        'severe': 'red',
        'high': 'purple',
        'low_collapse': 'green',
        'medium_collapse': 'orange',
        'severe_collapse': 'red',
        'high_collapse': 'purple',
    }

    for level, data in results_dict.items():
        steps = data.get('steps', list(range(len(data.get('values', [])))))
        values = data.get('values', [])

        color = colors.get(level, 'black')

        plt.plot(steps, values, label=level, color=color, linewidth=2)

    plt.xlabel('Training Steps')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Across Collapse Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
