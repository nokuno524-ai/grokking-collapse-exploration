import torch
import torch.nn as nn
from typing import List, Dict, Tuple

def slice_qkv_weight(
    in_proj_weight: torch.Tensor,
    d_model: int,
    n_heads: int,
    head_idx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given an in_proj_weight of shape (3 * d_model, d_model),
    extract the Q, K, V weights for a specific head.
    Returns (W_q, W_k, W_v) each of shape (d_head, d_model).
    """
    assert in_proj_weight.shape == (3 * d_model, d_model)
    d_head = d_model // n_heads

    q_weight = in_proj_weight[:d_model, :]
    k_weight = in_proj_weight[d_model:2*d_model, :]
    v_weight = in_proj_weight[2*d_model:, :]

    start = head_idx * d_head
    end = start + d_head

    return (
        q_weight[start:end, :],
        k_weight[start:end, :],
        v_weight[start:end, :]
    )

def inject_qkv_weight(
    base_in_proj_weight: torch.Tensor,
    donor_in_proj_weight: torch.Tensor,
    d_model: int,
    n_heads: int,
    head_idx: int
) -> torch.Tensor:
    """
    Replace the Q, K, V weights of head_idx in base_in_proj_weight
    with the corresponding weights from donor_in_proj_weight.
    Returns a new tensor.
    """
    d_head = d_model // n_heads
    start = head_idx * d_head
    end = start + d_head

    new_weight = base_in_proj_weight.clone()

    for offset in [0, d_model, 2*d_model]:
        new_weight[offset + start : offset + end, :] = donor_in_proj_weight[offset + start : offset + end, :]

    return new_weight

def inject_out_proj_weight(
    base_out_proj_weight: torch.Tensor,
    donor_out_proj_weight: torch.Tensor,
    d_model: int,
    n_heads: int,
    head_idx: int
) -> torch.Tensor:
    """
    Replace the output projection weights of head_idx in base_out_proj_weight
    (shape: d_model, d_model) with donor_out_proj_weight.
    Note: out_proj groups by heads on the input dimension (dim=1).
    Returns a new tensor.
    """
    d_head = d_model // n_heads
    start = head_idx * d_head
    end = start + d_head

    new_weight = base_out_proj_weight.clone()
    new_weight[:, start:end] = donor_out_proj_weight[:, start:end]

    return new_weight

def transplant_head(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    d_model: int = 128,
    n_heads: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Create a new state_dict with the specified head transplanted from donor to base.
    """
    new_sd = {k: v.clone() for k, v in base_sd.items()}

    in_proj_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
    out_proj_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"

    if in_proj_key in new_sd and in_proj_key in donor_sd:
        new_sd[in_proj_key] = inject_qkv_weight(
            new_sd[in_proj_key], donor_sd[in_proj_key], d_model, n_heads, head_idx
        )

    if out_proj_key in new_sd and out_proj_key in donor_sd:
        new_sd[out_proj_key] = inject_out_proj_weight(
            new_sd[out_proj_key], donor_sd[out_proj_key], d_model, n_heads, head_idx
        )

    # Handle biases if present
    in_proj_bias = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"
    if in_proj_bias in new_sd and in_proj_bias in donor_sd:
        # bias is (3 * d_model)
        new_bias = new_sd[in_proj_bias].clone()
        donor_b = donor_sd[in_proj_bias]
        d_head = d_model // n_heads
        start = head_idx * d_head
        end = start + d_head
        for offset in [0, d_model, 2*d_model]:
            new_bias[offset + start : offset + end] = donor_b[offset + start : offset + end]
        new_sd[in_proj_bias] = new_bias

    return new_sd
