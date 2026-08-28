"""
Surgical utilities for swapping or blending weights between transformer checkpoints.
"""

import torch
from typing import Dict, List, Set

def get_head_slice(d_model: int, n_heads: int, head_idx: int) -> slice:
    """Return slice for a specific head in a partitioned projection."""
    head_dim = d_model // n_heads
    return slice(head_idx * head_dim, (head_idx + 1) * head_dim)

def swap_weights(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    keys_to_swap: Set[str],
) -> Dict[str, torch.Tensor]:
    """
    Swap exact parameter keys from donor_sd into base_sd.
    Returns a new state_dict.
    """
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    for k in keys_to_swap:
        if k not in out_sd:
            raise KeyError(f"Key {k} not found in base state_dict.")
        if k not in donor_sd:
            raise KeyError(f"Key {k} not found in donor state_dict.")
        if out_sd[k].shape != donor_sd[k].shape:
            raise RuntimeError(f"Shape mismatch for {k}: {out_sd[k].shape} vs {donor_sd[k].shape}")
        out_sd[k] = donor_sd[k].clone()
    return out_sd

def blend_weights(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    keys_to_blend: Set[str],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """
    Linearly interpolate weights: (1 - alpha) * base + alpha * donor.
    Returns a new state_dict.
    """
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    for k in keys_to_blend:
        if k not in out_sd or k not in donor_sd:
            raise KeyError(f"Key {k} missing from base or donor.")
        if out_sd[k].shape != donor_sd[k].shape:
            raise RuntimeError(f"Shape mismatch for {k}.")
        out_sd[k] = (1.0 - alpha) * out_sd[k] + alpha * donor_sd[k]
    return out_sd

def swap_attention_head(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    d_model: int,
    n_heads: int,
) -> Dict[str, torch.Tensor]:
    """
    Swap a single attention head's weights (Q, K, V in in_proj, and out_proj)
    from donor to base.
    """
    out_sd = {k: v.clone() for k, v in base_sd.items()}

    in_proj_weight_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
    in_proj_bias_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"
    out_proj_weight_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"

    h_slice = get_head_slice(d_model, n_heads, head_idx)

    # in_proj groups Q, K, V along the first dimension (3 * d_model).
    # Each Q, K, V chunk is of size d_model.
    if in_proj_weight_key in base_sd and in_proj_weight_key in donor_sd:
        for offset in [0, d_model, 2 * d_model]:
            qkv_slice = slice(offset + h_slice.start, offset + h_slice.stop)
            out_sd[in_proj_weight_key][qkv_slice, :] = donor_sd[in_proj_weight_key][qkv_slice, :].clone()

    if in_proj_bias_key in base_sd and in_proj_bias_key in donor_sd:
        for offset in [0, d_model, 2 * d_model]:
            qkv_slice = slice(offset + h_slice.start, offset + h_slice.stop)
            out_sd[in_proj_bias_key][qkv_slice] = donor_sd[in_proj_bias_key][qkv_slice].clone()

    # out_proj_weight has shape (d_model, d_model), grouping heads along dim 1.
    if out_proj_weight_key in base_sd and out_proj_weight_key in donor_sd:
        out_sd[out_proj_weight_key][:, h_slice] = donor_sd[out_proj_weight_key][:, h_slice].clone()

    return out_sd

def swap_mlp(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """Swap the MLP (linear1 and linear2) weights for a given layer."""
    keys = {
        f"transformer.layers.{layer_idx}.linear1.weight",
        f"transformer.layers.{layer_idx}.linear1.bias",
        f"transformer.layers.{layer_idx}.linear2.weight",
        f"transformer.layers.{layer_idx}.linear2.bias",
    }
    # Only keep keys that actually exist (e.g. if bias=False)
    keys = {k for k in keys if k in base_sd}
    return swap_weights(base_sd, donor_sd, keys)

def swap_layer_norm(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """Swap the layer norm weights (norm1, norm2) for a given layer."""
    keys = {
        f"transformer.layers.{layer_idx}.norm1.weight",
        f"transformer.layers.{layer_idx}.norm1.bias",
        f"transformer.layers.{layer_idx}.norm2.weight",
        f"transformer.layers.{layer_idx}.norm2.bias",
    }
    keys = {k for k in keys if k in base_sd}
    return swap_weights(base_sd, donor_sd, keys)
