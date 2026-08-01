import torch
import torch.nn as nn
import scipy.stats
import numpy as np
from typing import Dict, Any, Tuple

def get_layer_norms(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Calculate L1, L2, Frobenius, and spectral norms for all parameters in the model.
    Returns a nested dictionary mapping parameter names to their norms.
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            p_detached = param.detach()
            layer_norms = {
                "l1": torch.linalg.norm(p_detached, ord=1).item(),
                "l2": torch.linalg.norm(p_detached, ord=2).item(),
                "frobenius": torch.linalg.norm(p_detached, ord='fro').item() if p_detached.dim() >= 2 else torch.linalg.norm(p_detached, ord=2).item(),
            }
            if p_detached.dim() == 2:
                # Spectral norm is the largest singular value, equivalent to ord=2 for matrices
                layer_norms["spectral"] = torch.linalg.matrix_norm(p_detached, ord=2).item()
            norms[name] = layer_norms
    return norms

def get_weight_distributions(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Calculate kurtosis and skewness for all parameters in the model.
    Returns a nested dictionary mapping parameter names to distribution metrics.
    """
    distributions = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            p_flat = param.detach().cpu().numpy().flatten()
            if len(p_flat) > 0:
                distributions[name] = {
                    "kurtosis": float(scipy.stats.kurtosis(p_flat)),
                    "skewness": float(scipy.stats.skew(p_flat)),
                }
    return distributions

def get_effective_ranks(model: nn.Module) -> Dict[str, float]:
    """
    Calculate the effective rank (SVD rank) of 2D weight matrices in the model.
    """
    ranks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() == 2:
            p_detached = param.detach()
            # Compute singular values
            s = torch.linalg.svdvals(p_detached)
            # Normalize to create a probability distribution over singular values
            s_norm = s / (s.sum() + 1e-10)
            # Compute entropy
            entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
            # Effective rank is exp(entropy)
            effective_rank = torch.exp(entropy).item()
            ranks[name] = effective_rank
    return ranks
