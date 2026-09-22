"""
Component transplant atlas toolkit for zero-shot testing.

This module provides tools for manipulating individual components (attention heads,
MLP neurons, whole layers) of a ModularArithmeticTransformer model.
It supports:
- Interventions: Transplantation (from a donor state dict), zero-ablation, mean-ablation
- Applying these interventions via modifying state dictionaries (for parameters).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Literal, Tuple, Any

# Types of interventions supported
InterventionType = Literal["transplant", "zero", "mean"]
ComponentType = Literal["head", "neuron", "layer"]

def get_head_indices(head_idx: int, d_model: int, n_heads: int) -> Tuple[int, int]:
    """Calculate the start and end indices of a given attention head in concatenated projections."""
    d_head = d_model // n_heads
    start = head_idx * d_head
    end = (head_idx + 1) * d_head
    return start, end

def apply_intervention_to_tensor(
    tensor: torch.Tensor,
    donor_tensor: Optional[torch.Tensor],
    intervention: InterventionType,
    dim: int,
    indices: Tuple[int, int]
) -> torch.Tensor:
    """Apply an intervention to a specific slice of a tensor.

    Args:
        tensor: The original tensor to modify.
        donor_tensor: The tensor to copy from (used if intervention == "transplant").
        intervention: The type of intervention ("transplant", "zero", "mean").
        dim: The dimension along which to slice.
        indices: A tuple (start, end) defining the slice.

    Returns:
        The modified tensor.
    """
    start, end = indices

    # We must clone to avoid modifying the original
    new_tensor = tensor.clone()

    # Create slicing logic based on dimension
    slices = [slice(None)] * new_tensor.ndim
    slices[dim] = slice(start, end)

    if intervention == "transplant":
        if donor_tensor is None:
            raise ValueError("donor_tensor must be provided for transplant intervention")
        new_tensor[tuple(slices)] = donor_tensor[tuple(slices)].clone()
    elif intervention == "zero":
        new_tensor[tuple(slices)] = 0.0
    elif intervention == "mean":
        # Mean across the specific dimension being modified
        mean_val = new_tensor[tuple(slices)].mean()
        new_tensor[tuple(slices)] = mean_val
    else:
        raise ValueError(f"Unknown intervention type: {intervention}")

    return new_tensor

def patch_attention_head(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Optional[Dict[str, torch.Tensor]],
    layer_idx: int,
    head_idx: int,
    d_model: int,
    n_heads: int,
    intervention: InterventionType = "transplant"
) -> Dict[str, torch.Tensor]:
    """Apply an intervention to a single attention head within a specific layer.

    Note: The `in_proj_weight` is concatenated (q, k, v), so we must apply the
    intervention to all three projections for the given head.

    Args:
        base_sd: The original state dict.
        donor_sd: The donor state dict (optional if not transplanting).
        layer_idx: The index of the transformer layer.
        head_idx: The index of the attention head (0-indexed).
        d_model: The embedding dimension.
        n_heads: The number of attention heads.
        intervention: The type of intervention ("transplant", "zero", "mean").

    Returns:
        A new state dict with the intervention applied.
    """
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    start, end = get_head_indices(head_idx, d_model, n_heads)

    prefix = f"transformer.layers.{layer_idx}.self_attn"

    # Process in_proj_weight/bias (which are concatenated Q, K, V)
    for param_name in ["in_proj_weight", "in_proj_bias"]:
        key = f"{prefix}.{param_name}"
        if key in out_sd:
            base_tensor = out_sd[key]
            donor_tensor = donor_sd[key] if donor_sd is not None and key in donor_sd else None

            # For concatenated projections, apply to Q, K, and V parts
            # Shape is (3 * d_model, ...)
            for part in range(3):
                part_start = part * d_model + start
                part_end = part * d_model + end

                out_sd[key] = apply_intervention_to_tensor(
                    out_sd[key],
                    donor_tensor,
                    intervention,
                    dim=0, # Output dimension is always 0
                    indices=(part_start, part_end)
                )

    # Process out_proj.weight/bias
    out_proj_w_key = f"{prefix}.out_proj.weight"
    if out_proj_w_key in out_sd:
        donor_tensor = donor_sd[out_proj_w_key] if donor_sd is not None else None
        # out_proj.weight has shape (d_model, d_model), the input dim is 1
        out_sd[out_proj_w_key] = apply_intervention_to_tensor(
            out_sd[out_proj_w_key],
            donor_tensor,
            intervention,
            dim=1,
            indices=(start, end)
        )

    # We do NOT patch out_proj.bias because bias applies after summing across heads.
    # We can't trivially attribute a slice of the bias to a single head.

    return out_sd

def patch_mlp_neuron(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Optional[Dict[str, torch.Tensor]],
    layer_idx: int,
    neuron_idx: int,
    intervention: InterventionType = "transplant"
) -> Dict[str, torch.Tensor]:
    """Apply an intervention to a single MLP neuron within a specific layer.

    Args:
        base_sd: The original state dict.
        donor_sd: The donor state dict.
        layer_idx: The index of the transformer layer.
        neuron_idx: The index of the MLP neuron (0-indexed).
        intervention: The type of intervention.

    Returns:
        A new state dict.
    """
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    prefix = f"transformer.layers.{layer_idx}"

    # Linear 1 (d_model -> d_ff): the neuron is an output channel (dim=0)
    w1_key = f"{prefix}.linear1.weight"
    b1_key = f"{prefix}.linear1.bias"

    if w1_key in out_sd:
        d_tensor = donor_sd[w1_key] if donor_sd is not None else None
        out_sd[w1_key] = apply_intervention_to_tensor(
            out_sd[w1_key], d_tensor, intervention, dim=0, indices=(neuron_idx, neuron_idx+1)
        )

    if b1_key in out_sd:
        d_tensor = donor_sd[b1_key] if donor_sd is not None else None
        out_sd[b1_key] = apply_intervention_to_tensor(
            out_sd[b1_key], d_tensor, intervention, dim=0, indices=(neuron_idx, neuron_idx+1)
        )

    # Linear 2 (d_ff -> d_model): the neuron is an input channel (dim=1)
    w2_key = f"{prefix}.linear2.weight"
    if w2_key in out_sd:
        d_tensor = donor_sd[w2_key] if donor_sd is not None else None
        out_sd[w2_key] = apply_intervention_to_tensor(
            out_sd[w2_key], d_tensor, intervention, dim=1, indices=(neuron_idx, neuron_idx+1)
        )

    # Again, linear2.bias cannot be trivially attributed to a single input neuron

    return out_sd

def patch_layer(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Optional[Dict[str, torch.Tensor]],
    layer_idx: int,
    intervention: InterventionType = "transplant"
) -> Dict[str, torch.Tensor]:
    """Apply an intervention to an entire layer.

    Args:
        base_sd: The original state dict.
        donor_sd: The donor state dict.
        layer_idx: The index of the transformer layer.
        intervention: The type of intervention.

    Returns:
        A new state dict.
    """
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    prefix = f"transformer.layers.{layer_idx}."

    for key in out_sd.keys():
        if key.startswith(prefix):
            if intervention == "transplant":
                if donor_sd is None or key not in donor_sd:
                    continue
                out_sd[key] = donor_sd[key].clone()
            elif intervention == "zero":
                out_sd[key] = torch.zeros_like(out_sd[key])
            elif intervention == "mean":
                out_sd[key] = torch.full_like(out_sd[key], out_sd[key].mean().item())

    return out_sd
