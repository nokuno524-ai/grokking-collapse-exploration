import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import re

def _get_qkv_head_indices(n_heads: int, head_idx: int, d_model: int) -> torch.Tensor:
    head_dim = d_model // n_heads
    start = head_idx * head_dim
    end = start + head_dim
    return torch.arange(start, end)

def _get_in_proj_head_indices(n_heads: int, head_idx: int, d_model: int) -> torch.Tensor:
    head_dim = d_model // n_heads
    q_start = head_idx * head_dim
    q_end = q_start + head_dim
    k_start = d_model + head_idx * head_dim
    k_end = k_start + head_dim
    v_start = 2 * d_model + head_idx * head_dim
    v_end = v_start + head_dim
    return torch.cat([
        torch.arange(q_start, q_end),
        torch.arange(k_start, k_end),
        torch.arange(v_start, v_end)
    ])

def swap_attention_head(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    n_heads: int,
    d_model: int
) -> Dict[str, torch.Tensor]:
    """Transplants a specific attention head from donor_sd to base_sd."""
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    prefix = f"transformer.layers.{layer_idx}.self_attn"

    in_proj_weight_key = f"{prefix}.in_proj_weight"
    in_proj_bias_key = f"{prefix}.in_proj_bias"
    out_proj_weight_key = f"{prefix}.out_proj.weight"
    # out_proj bias is shared across heads, so we typically don't swap it for a single head

    in_idx = _get_in_proj_head_indices(n_heads, head_idx, d_model).to(base_sd[in_proj_weight_key].device)
    out_idx = _get_qkv_head_indices(n_heads, head_idx, d_model).to(base_sd[out_proj_weight_key].device)

    # in_proj_weight: (3 * d_model, d_model) -> swap row slices
    if in_proj_weight_key in donor_sd:
        out_sd[in_proj_weight_key][in_idx, :] = donor_sd[in_proj_weight_key][in_idx, :].clone().to(out_sd[in_proj_weight_key].device)
    if in_proj_bias_key in donor_sd and in_proj_bias_key in base_sd:
        out_sd[in_proj_bias_key][in_idx] = donor_sd[in_proj_bias_key][in_idx].clone().to(out_sd[in_proj_bias_key].device)

    # out_proj.weight: (d_model, d_model) -> swap column slices
    if out_proj_weight_key in donor_sd:
        out_sd[out_proj_weight_key][:, out_idx] = donor_sd[out_proj_weight_key][:, out_idx].clone().to(out_sd[out_proj_weight_key].device)

    return out_sd

def swap_mlp(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int
) -> Dict[str, torch.Tensor]:
    """Transplants the MLP (linear1, linear2) of a specific layer."""
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    prefix = f"transformer.layers.{layer_idx}."

    for key in base_sd.keys():
        if (key.startswith(f"{prefix}linear1.") or key.startswith(f"{prefix}linear2.")) and key in donor_sd:
            if base_sd[key].shape != donor_sd[key].shape:
                raise ValueError(f"Shape mismatch for {key}")
            out_sd[key] = donor_sd[key].clone().to(out_sd[key].device)

    return out_sd

def swap_layernorm(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    norm_idx: int = 1
) -> Dict[str, torch.Tensor]:
    """Transplants LayerNorm 1 or 2 of a specific layer."""
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    prefix = f"transformer.layers.{layer_idx}.norm{norm_idx}."

    for key in base_sd.keys():
        if key.startswith(prefix) and key in donor_sd:
            if base_sd[key].shape != donor_sd[key].shape:
                raise ValueError(f"Shape mismatch for {key}")
            out_sd[key] = donor_sd[key].clone().to(out_sd[key].device)

    return out_sd
