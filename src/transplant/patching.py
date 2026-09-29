import re
import torch
from typing import Dict, Optional


def keys_for(component: str, sd: Dict[str, torch.Tensor]) -> list[str]:
    """Get all state_dict keys that match the given component regex pattern."""
    COMPONENT_PATTERNS: Dict[str, str] = {
        "token_embed": r"^token_embed\.",
        "pos_embed": r"^pos_embed\.",
        "self_attn_in_proj": r"^transformer\.layers\.\d+\.self_attn\.in_proj_(weight|bias)$",
        "self_attn_out_proj": r"^transformer\.layers\.\d+\.self_attn\.out_proj\.",
        "linear1": r"^transformer\.layers\.\d+\.linear1\.",
        "linear2": r"^transformer\.layers\.\d+\.linear2\.",
        "norm1": r"^transformer\.layers\.\d+\.norm1\.",
        "norm2": r"^transformer\.layers\.\d+\.norm2\.",
        "ln": r"^ln\.",
        "output_head": r"^output_head\.",
    }
    # Allow passing custom regex directly if not in predefined map
    pat = COMPONENT_PATTERNS.get(component, component)
    return [k for k in sd.keys() if re.match(pat, k)]


def patch_head_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    d_model: int,
    n_heads: int
) -> Dict[str, torch.Tensor]:
    """
    Surgically graft a specific attention head from donor_sd into base_sd.

    Args:
        base_sd: Recipient state dictionary.
        donor_sd: Source state dictionary.
        layer_idx: Transformer layer index.
        head_idx: Attention head index.
        d_model: Total embedding dimension.
        n_heads: Total number of attention heads.

    Returns:
        A new state_dict with the head transplanted.
    """
    out = {k: v.clone() for k, v in base_sd.items()}
    d_head = d_model // n_heads
    start_idx = head_idx * d_head
    end_idx = (head_idx + 1) * d_head

    # QKV fused weights
    in_proj_w = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
    in_proj_b = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"

    if in_proj_w in base_sd and in_proj_w in donor_sd:
        # in_proj_weight is shape (3 * d_model, d_model)
        # We need to slice Q, K, V separately
        for offset in [0, d_model, 2 * d_model]:
            out[in_proj_w][offset + start_idx : offset + end_idx, :] = donor_sd[in_proj_w][offset + start_idx : offset + end_idx, :].clone().to(dtype=out[in_proj_w].dtype, device=out[in_proj_w].device)

    if in_proj_b in base_sd and in_proj_b in donor_sd:
        for offset in [0, d_model, 2 * d_model]:
            out[in_proj_b][offset + start_idx : offset + end_idx] = donor_sd[in_proj_b][offset + start_idx : offset + end_idx].clone().to(dtype=out[in_proj_b].dtype, device=out[in_proj_b].device)

    # Output projection
    out_proj_w = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"
    if out_proj_w in base_sd and out_proj_w in donor_sd:
        # out_proj_weight is shape (d_model, d_model)
        # Heads are concatenated along the input dimension for out_proj
        out[out_proj_w][:, start_idx : end_idx] = donor_sd[out_proj_w][:, start_idx : end_idx].clone().to(dtype=out[out_proj_w].dtype, device=out[out_proj_w].device)

    return out


def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
    key_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Graft a complete component (defined by regex) from donor_sd to base_sd.

    Args:
        base_sd: Recipient state dictionary.
        donor_sd: Source state dictionary.
        component: The component name or regex.
        key_mapping: Optional dict to map donor keys to base keys.

    Returns:
        A new state_dict with the component transplanted.
    """
    out = {k: v.clone() for k, v in base_sd.items()}
    keys_to_patch = keys_for(component, base_sd)

    for base_k in keys_to_patch:
        donor_k = base_k
        if key_mapping and base_k in key_mapping:
            donor_k = key_mapping[base_k]

        if donor_k in donor_sd:
            donor_t = donor_sd[donor_k]
            base_t = base_sd[base_k]

            if donor_t.shape != base_t.shape:
                raise ValueError(
                    f"shape mismatch on {base_k}: donor {tuple(donor_t.shape)} "
                    f"vs base {tuple(base_t.shape)}"
                )
            # Explicitly cast to recipient dtype and device to avoid hardware mismatch
            out[base_k] = donor_t.clone().to(dtype=base_t.dtype, device=base_t.device)

    return out
