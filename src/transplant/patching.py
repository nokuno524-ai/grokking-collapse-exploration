import re
import torch
from typing import Dict, List, Optional, Callable

# Standard component regex patterns for our transformer architecture
COMPONENT_PATTERNS = {
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
    "mlp": r"^transformer\.layers\.\d+\.(linear1|linear2)\.",
    "attention": r"^transformer\.layers\.\d+\.self_attn\.",
}

def get_keys_for_component(component: str, state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Return all keys in state_dict that match the given component pattern."""
    if component not in COMPONENT_PATTERNS:
        raise ValueError(f"Unknown component: {component}. Available: {list(COMPONENT_PATTERNS.keys())}")
    pattern = COMPONENT_PATTERNS[component]
    return [k for k in state_dict.keys() if re.match(pattern, k)]

def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    components: List[str],
    key_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Patch specific components from donor_sd into base_sd.
    Returns a new state dictionary.
    """
    patched_sd = {k: v.clone() for k, v in base_sd.items()}

    for component in components:
        keys = get_keys_for_component(component, base_sd)
        if not keys:
            print(f"Warning: No keys found for component {component}")

        for key in keys:
            donor_key = key
            if key_mapping and key in key_mapping:
                donor_key = key_mapping[key]

            if donor_key not in donor_sd:
                raise KeyError(f"Key {donor_key} (mapped from {key}) not found in donor_sd")

            # Check shapes
            if base_sd[key].shape != donor_sd[donor_key].shape:
                raise ValueError(
                    f"Shape mismatch for key {key}: base is {base_sd[key].shape}, "
                    f"donor is {donor_sd[donor_key].shape}"
                )

            # Copy keeping the device and dtype of the recipient
            dtype, device = base_sd[key].dtype, base_sd[key].device
            patched_sd[key] = donor_sd[donor_key].clone().to(dtype=dtype, device=device)

    return patched_sd

def patch_head_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    d_model: int,
    n_heads: int,
) -> Dict[str, torch.Tensor]:
    """
    Patch a single attention head from donor_sd to base_sd.
    Handles fused QKV slicing correctly.
    """
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_head = d_model // n_heads
    start_idx = head_idx * d_head
    end_idx = (head_idx + 1) * d_head

    patched_sd = {k: v.clone() for k, v in base_sd.items()}

    # In-proj (fused QKV)
    in_weight_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
    in_bias_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"

    if in_weight_key in base_sd and in_weight_key in donor_sd:
        # q, k, v are stacked
        for offset in [0, d_model, 2 * d_model]:
            s, e = offset + start_idx, offset + end_idx
            patched_sd[in_weight_key][s:e, :] = donor_sd[in_weight_key][s:e, :].clone().to(
                dtype=base_sd[in_weight_key].dtype, device=base_sd[in_weight_key].device
            )

    if in_bias_key in base_sd and in_bias_key in donor_sd:
        for offset in [0, d_model, 2 * d_model]:
            s, e = offset + start_idx, offset + end_idx
            patched_sd[in_bias_key][s:e] = donor_sd[in_bias_key][s:e].clone().to(
                dtype=base_sd[in_bias_key].dtype, device=base_sd[in_bias_key].device
            )

    # Out-proj (weight only, bias is shared)
    out_weight_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"
    if out_weight_key in base_sd and out_weight_key in donor_sd:
        patched_sd[out_weight_key][:, start_idx:end_idx] = donor_sd[out_weight_key][:, start_idx:end_idx].clone().to(
            dtype=base_sd[out_weight_key].dtype, device=base_sd[out_weight_key].device
        )

    return patched_sd
