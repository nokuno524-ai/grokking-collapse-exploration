"""
Utilities for targeted causal ablation (zeroing, mean-ablating, resampling)
of neural network components in state dictionaries.
"""

from typing import Dict, Optional
import torch

from .patching import keys_for

def zero_ablate_component(
    base_sd: Dict[str, torch.Tensor],
    component: str,
) -> Dict[str, torch.Tensor]:
    """
    Return a new state dictionary with the specified component zeroed out.

    Args:
        base_sd: The recipient state dictionary.
        component: The component name pattern to zero out.

    Returns:
        A new state dictionary with the component zeroed.
    """
    out = {k: v.clone() for k, v in base_sd.items()}
    for k in keys_for(component, base_sd):
        out[k].zero_()
    return out

def mean_ablate_component(
    base_sd: Dict[str, torch.Tensor],
    component: str,
    dataset_mean_sd: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Return a new state dictionary with the specified component replaced by its mean.
    If dataset_mean_sd is provided, uses those means. Otherwise, uses the mean of
    the weight matrix itself (which is a rough approximation of mean ablation).

    Args:
        base_sd: The recipient state dictionary.
        component: The component name pattern to mean-ablate.
        dataset_mean_sd: Optional state dict containing the mean weights over a dataset.

    Returns:
        A new state dictionary.
    """
    out = {k: v.clone() for k, v in base_sd.items()}
    for k in keys_for(component, base_sd):
        if dataset_mean_sd is not None and k in dataset_mean_sd:
            if dataset_mean_sd[k].shape != base_sd[k].shape:
                raise ValueError(f"Shape mismatch for {k}")
            out[k] = dataset_mean_sd[k].clone()
        else:
            # Fallback: mean over the entire tensor
            mean_val = base_sd[k].mean()
            out[k].fill_(mean_val)
    return out

def resample_ablate_component(
    base_sd: Dict[str, torch.Tensor],
    component: str,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Return a new state dictionary with the specified component randomly permuted.
    This preserves the empirical distribution of the weights but destroys spatial
    correlations/structure.

    Args:
        base_sd: The recipient state dictionary.
        component: The component name pattern to resample.
        rng: Optional random generator.

    Returns:
        A new state dictionary.
    """
    out = {k: v.clone() for k, v in base_sd.items()}
    if rng is None:
        rng = torch.Generator().manual_seed(0)

    for k in keys_for(component, base_sd):
        w = base_sd[k]
        flat_w = w.flatten()
        idx = torch.randperm(flat_w.numel(), generator=rng, device=w.device)
        out[k] = flat_w[idx].reshape(w.shape)

    return out
