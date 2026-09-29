import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable
import numpy as np

def zero_ablate_component(
    state_dict: Dict[str, torch.Tensor],
    keys: List[str]
) -> Dict[str, torch.Tensor]:
    """Zero out specified keys in the state dict."""
    out = {k: v.clone() for k, v in state_dict.items()}
    for key in keys:
        if key in out:
            out[key] = torch.zeros_like(out[key])
    return out

def mean_ablate_component(
    state_dict: Dict[str, torch.Tensor],
    keys: List[str]
) -> Dict[str, torch.Tensor]:
    """Mean ablate specified keys in the state dict."""
    out = {k: v.clone() for k, v in state_dict.items()}
    for key in keys:
        if key in out:
            mean_val = out[key].mean()
            out[key] = torch.full_like(out[key], mean_val)
    return out

def resample_ablate_component(
    state_dict: Dict[str, torch.Tensor],
    keys: List[str],
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """Resample weights from a normal distribution with the same mean and std."""
    out = {k: v.clone() for k, v in state_dict.items()}
    rng = torch.Generator(device=out[next(iter(out.keys()))].device)
    rng.manual_seed(seed)

    for key in keys:
        if key in out:
            tensor = out[key]
            mean = tensor.mean().item()
            std = tensor.std().item() if tensor.numel() > 1 else 1.0
            out[key] = torch.normal(mean=mean, std=std, size=tensor.size(), generator=rng, dtype=tensor.dtype, device=tensor.device)
    return out
