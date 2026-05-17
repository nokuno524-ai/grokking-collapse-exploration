import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
import scipy.linalg

def get_weight_norms(model: nn.Module) -> Dict[str, float]:
    """
    Calculate the L2 norm of the weights for each named parameter in the model.
    Also computes the 'total' L2 norm.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping parameter names to their L2 norm, including 'total'.
    """
    norms = {}
    total_squared_norm = 0.0
    for name, param in model.named_parameters():
        norm = param.data.norm(2).item()
        norms[name] = norm
        total_squared_norm += norm ** 2
    norms['total'] = total_squared_norm ** 0.5
    return norms

def get_effective_rank(tensor: torch.Tensor, threshold: float = 1e-6) -> float:
    """
    Calculate the effective rank of a 2D weight matrix based on its singular value distribution.
    Effective rank is computed as exp(Shannon entropy of normalized singular values).

    Args:
        tensor: 2D PyTorch tensor.
        threshold: Small value to avoid log(0).

    Returns:
        Effective rank as a float. Returns 0 for 1D or 0D tensors.
    """
    if tensor.dim() < 2:
        return 0.0

    # If more than 2D, flatten all but the first dimension (like nn.Linear does)
    if tensor.dim() > 2:
        tensor = tensor.view(tensor.size(0), -1)

    s = torch.linalg.svdvals(tensor)
    s_sum = s.sum()
    if s_sum <= threshold:
        return 0.0

    p = s / s_sum
    entropy = -torch.sum(p * torch.log(p + threshold))
    return torch.exp(entropy).item()

def get_layer_effective_ranks(model: nn.Module) -> Dict[str, float]:
    """
    Compute effective rank for all 2D+ weight matrices in the model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping parameter names to their effective rank.
    """
    ranks = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            ranks[name] = get_effective_rank(param.data)
    return ranks

def get_singular_value_distribution(tensor: torch.Tensor) -> np.ndarray:
    """
    Get the singular values of a 2D weight matrix.

    Args:
        tensor: 2D PyTorch tensor.

    Returns:
        Numpy array of singular values sorted in descending order.
    """
    if tensor.dim() < 2:
        return np.array([])

    if tensor.dim() > 2:
        tensor = tensor.view(tensor.size(0), -1)

    s = torch.linalg.svdvals(tensor)
    return s.cpu().numpy()

def calculate_weight_velocity(model_t1: nn.Module, model_t2: nn.Module) -> Dict[str, float]:
    """
    Calculate the L2 distance between parameters of two models with the same architecture,
    representing the parameter movement (velocity) between two timesteps.

    Args:
        model_t1: Model at timestep 1.
        model_t2: Model at timestep 2.

    Returns:
        Dictionary mapping parameter names to their L2 movement, including 'total'.
    """
    velocities = {}
    total_squared_diff = 0.0

    params_t1 = dict(model_t1.named_parameters())
    params_t2 = dict(model_t2.named_parameters())

    for name in params_t1.keys():
        if name in params_t2:
            diff = (params_t1[name].data - params_t2[name].data).norm(2).item()
            velocities[name] = diff
            total_squared_diff += diff ** 2

    velocities['total'] = total_squared_diff ** 0.5
    return velocities
