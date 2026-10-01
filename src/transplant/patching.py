"""
Core utilities for patching transformer state dictionaries.
"""

from typing import Dict, Optional, List
import re
import torch

from src.transplant_rescue import COMPONENT_PATTERNS, random_basis_swap

def keys_for(component: str, sd: Dict[str, torch.Tensor]) -> List[str]:
    """Return all keys in the state dict that match the given component pattern."""
    if component not in COMPONENT_PATTERNS:
        raise ValueError(f"unknown component {component!r}; valid: {list(COMPONENT_PATTERNS)}")
    pat = COMPONENT_PATTERNS[component]
    return [k for k in sd.keys() if re.match(pat, k)]

def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
    randomize: bool = False,
    rng: Optional[torch.Generator] = None,
    key_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Return a new state_dict = base, with `component` replaced by donor's.

    Args:
        base_sd: The recipient state dictionary.
        donor_sd: The source state dictionary.
        component: The component name to patch.
        randomize: If True, replace with a random-orthonormal-basis version of base
                   (specificity control); donor_sd is unused.
        rng: Random generator for randomize.
        key_mapping: Optional dict to map keys from donor to base, if names differ.

    Returns:
        A new state dictionary with the patched component.
    """
    out = {k: v.clone() for k, v in base_sd.items()}
    if randomize:
        if rng is None:
            rng = torch.Generator().manual_seed(0)
        for k in keys_for(component, base_sd):
            out[k] = random_basis_swap(base_sd[k], rng)
    else:
        for k in keys_for(component, base_sd):
            donor_k = key_mapping.get(k, k) if key_mapping else k
            if donor_k not in donor_sd:
                continue
            if donor_sd[donor_k].shape != base_sd[k].shape:
                raise ValueError(
                    f"shape mismatch on {k}: donor {tuple(donor_sd[donor_k].shape)} "
                    f"vs base {tuple(base_sd[k].shape)}"
                )
            out[k] = donor_sd[donor_k].clone()
    return out

def patch_head_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    d_model: int,
    n_heads: int,
) -> Dict[str, torch.Tensor]:
    """
    Surgically slice and replace a single attention head's weights from donor to base.

    Args:
        base_sd: The recipient state dictionary.
        donor_sd: The source state dictionary.
        layer_idx: The transformer layer index.
        head_idx: The attention head index to patch.
        d_model: The model dimension.
        n_heads: The number of attention heads.

    Returns:
        A new state dictionary with the patched head.
    """
    if d_model % n_heads != 0:
        raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")

    d_head = d_model // n_heads
    out = {k: v.clone() for k, v in base_sd.items()}

    start = head_idx * d_head
    end = (head_idx + 1) * d_head

    # Prefix for PyTorch's nn.MultiheadAttention inside nn.TransformerEncoderLayer
    prefix = f"transformer.layers.{layer_idx}.self_attn."

    # 1. in_proj_weight (shape: 3 * d_model, d_model) -> fused QKV
    # Each of Q, K, V has shape (d_model, d_model), concatenated along dim 0.
    # Within each, the head is sliced along dim 0.
    in_proj_w = f"{prefix}in_proj_weight"
    if in_proj_w in base_sd and in_proj_w in donor_sd:
        for offset in [0, d_model, 2 * d_model]:
            out[in_proj_w][offset + start : offset + end, :] = \
                donor_sd[in_proj_w][offset + start : offset + end, :].clone()

    # 2. in_proj_bias (shape: 3 * d_model)
    in_proj_b = f"{prefix}in_proj_bias"
    if in_proj_b in base_sd and in_proj_b in donor_sd:
        for offset in [0, d_model, 2 * d_model]:
            out[in_proj_b][offset + start : offset + end] = \
                donor_sd[in_proj_b][offset + start : offset + end].clone()

    # 3. out_proj.weight (shape: d_model, d_model)
    # The heads are concatenated along the input dimension (dim 1).
    out_proj_w = f"{prefix}out_proj.weight"
    if out_proj_w in base_sd and out_proj_w in donor_sd:
        out[out_proj_w][:, start : end] = \
            donor_sd[out_proj_w][:, start : end].clone()

    # Note: out_proj.bias is shared across all heads, so we don't patch it per-head.

    return out
