import torch
import re
from typing import Dict, List, Optional
from src.transplant.transplant_rescue import keys_for, COMPONENT_PATTERNS, random_basis_swap

def patch_head_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    d_head: int,
    n_heads: int,
    randomize: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Return a new state_dict with a specific attention head replaced by donor's.

    Args:
        base_sd: The recipient state dictionary.
        donor_sd: The donor state dictionary.
        layer_idx: The index of the transformer layer.
        head_idx: The index of the attention head to patch (0 to n_heads - 1).
        d_head: The dimension of each attention head (d_model // n_heads).
        n_heads: The total number of attention heads.
        randomize: If True, replace with a random-orthonormal-basis version of base.
        rng: Random number generator.
    """
    out = {k: v.clone() for k, v in base_sd.items()}

    in_proj_weight_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
    in_proj_bias_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"
    out_proj_weight_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"

    if in_proj_weight_key not in base_sd or out_proj_weight_key not in base_sd:
        return out

    start_idx = head_idx * d_head
    end_idx = (head_idx + 1) * d_head

    d_model = n_heads * d_head

    # in_proj_weight/bias slices (Q, K, V are concatenated)
    for qkv_offset in [0, d_model, 2 * d_model]:
        slice_start = qkv_offset + start_idx
        slice_end = qkv_offset + end_idx

        if randomize:
            if rng is None:
                rng = torch.Generator().manual_seed(0)
            if in_proj_weight_key in base_sd:
                out[in_proj_weight_key][slice_start:slice_end] = random_basis_swap(base_sd[in_proj_weight_key][slice_start:slice_end], rng)
            if in_proj_bias_key in base_sd:
                out[in_proj_bias_key][slice_start:slice_end] = random_basis_swap(base_sd[in_proj_bias_key][slice_start:slice_end], rng)
        else:
            if in_proj_weight_key in base_sd and in_proj_weight_key in donor_sd:
                out[in_proj_weight_key][slice_start:slice_end] = donor_sd[in_proj_weight_key][slice_start:slice_end].clone().to(dtype=base_sd[in_proj_weight_key].dtype, device=base_sd[in_proj_weight_key].device)
            if in_proj_bias_key in base_sd and in_proj_bias_key in donor_sd:
                out[in_proj_bias_key][slice_start:slice_end] = donor_sd[in_proj_bias_key][slice_start:slice_end].clone().to(dtype=base_sd[in_proj_bias_key].dtype, device=base_sd[in_proj_bias_key].device)

    # out_proj_weight slice (input dimension to out_proj is partitioned by head)
    if randomize:
        if rng is None:
            rng = torch.Generator().manual_seed(0)
        if out_proj_weight_key in base_sd:
            out[out_proj_weight_key][:, start_idx:end_idx] = random_basis_swap(base_sd[out_proj_weight_key][:, start_idx:end_idx], rng)
    else:
        if out_proj_weight_key in base_sd and out_proj_weight_key in donor_sd:
            out[out_proj_weight_key][:, start_idx:end_idx] = donor_sd[out_proj_weight_key][:, start_idx:end_idx].clone().to(dtype=base_sd[out_proj_weight_key].dtype, device=base_sd[out_proj_weight_key].device)

    return out


def patch_layer_component_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    component_name: str,
    randomize: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Return a new state_dict with a specific layer component replaced by donor's.

    Args:
        base_sd: The recipient state dictionary.
        donor_sd: The donor state dictionary.
        layer_idx: The index of the transformer layer.
        component_name: The name of the component (e.g., 'linear1', 'linear2', 'self_attn').
        randomize: If True, replace with a random-orthonormal-basis version of base.
        rng: Random number generator.
    """
    out = {k: v.clone() for k, v in base_sd.items()}

    # Ensure component name is valid and get regex
    if component_name in COMPONENT_PATTERNS:
        # Standard component like linear1
        pat = COMPONENT_PATTERNS[component_name]
    else:
        # Custom component name for specific layer, e.g., 'mlp' -> linear1 and linear2
        if component_name == 'mlp':
            pat = r"^transformer\.layers\.\d+\.linear[12]\."
        elif component_name == 'self_attn':
            pat = r"^transformer\.layers\.\d+\.self_attn\."
        else:
            pat = r"^transformer\.layers\.\d+\." + re.escape(component_name) + r"\."

    # We want to match only the specific layer
    pat = pat.replace(r"\d+", str(layer_idx))

    keys_to_patch = [k for k in base_sd.keys() if re.match(pat, k)]

    for k in keys_to_patch:
        if randomize:
            if rng is None:
                rng = torch.Generator().manual_seed(0)
            out[k] = random_basis_swap(base_sd[k], rng)
        else:
            if k not in donor_sd:
                continue
            if donor_sd[k].shape != base_sd[k].shape:
                raise ValueError(
                    f"shape mismatch on {k}: donor {tuple(donor_sd[k].shape)} "
                    f"vs base {tuple(base_sd[k].shape)}"
                )
            out[k] = donor_sd[k].clone().to(dtype=base_sd[k].dtype, device=base_sd[k].device)

    return out
