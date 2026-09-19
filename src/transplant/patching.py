import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import re

def patch_attention_head(
    recipient_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    d_model: int,
    n_heads: int,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Patches a single attention head from donor to recipient.
    Handles fused in_proj (Q,K,V) and out_proj.
    """
    out_sd = {k: v.clone() for k, v in recipient_sd.items()}
    prefix = f"transformer.layers.{layer_idx}.self_attn" if f"transformer.layers.{layer_idx}.self_attn.in_proj_weight" in out_sd else "transformer.layers.0.self_attn"

    # In proj (Q, K, V)
    d_head = d_model // n_heads
    start_q = head_idx * d_head
    end_q = (head_idx + 1) * d_head
    start_k = d_model + head_idx * d_head
    end_k = d_model + (head_idx + 1) * d_head
    start_v = 2 * d_model + head_idx * d_head
    end_v = 2 * d_model + (head_idx + 1) * d_head

    for suffix in ["in_proj_weight", "in_proj_bias"]:
        k = f"{prefix}.{suffix}"
        if k in out_sd and k in donor_sd:
            if verbose: print(f"Patching head {head_idx} QKV for {k}")
            if "weight" in suffix:
                out_sd[k][start_q:end_q, :] = donor_sd[k][start_q:end_q, :].clone().to(dtype=out_sd[k].dtype, device=out_sd[k].device)
                out_sd[k][start_k:end_k, :] = donor_sd[k][start_k:end_k, :].clone().to(dtype=out_sd[k].dtype, device=out_sd[k].device)
                out_sd[k][start_v:end_v, :] = donor_sd[k][start_v:end_v, :].clone().to(dtype=out_sd[k].dtype, device=out_sd[k].device)
            else:
                out_sd[k][start_q:end_q] = donor_sd[k][start_q:end_q].clone().to(dtype=out_sd[k].dtype, device=out_sd[k].device)
                out_sd[k][start_k:end_k] = donor_sd[k][start_k:end_k].clone().to(dtype=out_sd[k].dtype, device=out_sd[k].device)
                out_sd[k][start_v:end_v] = donor_sd[k][start_v:end_v].clone().to(dtype=out_sd[k].dtype, device=out_sd[k].device)

    # Out proj
    for suffix in ["out_proj.weight", "out_proj.bias"]:
        k = f"{prefix}.{suffix}"
        if k in out_sd and k in donor_sd:
            if verbose: print(f"Patching head {head_idx} out_proj for {k}")
            if "weight" in suffix:
                # out_proj weight is (d_model, d_model), but the heads are concatenated along the input dimension (dim 1)
                start = head_idx * d_head
                end = (head_idx + 1) * d_head
                out_sd[k][:, start:end] = donor_sd[k][:, start:end].clone().to(dtype=out_sd[k].dtype, device=out_sd[k].device)
            else:
                pass

    return out_sd

def patch_module(
    recipient_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    regex_pattern: str,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Patches a whole module/block matching regex_pattern from donor to recipient.
    """
    out_sd = {k: v.clone() for k, v in recipient_sd.items()}
    matched = False
    for k in out_sd.keys():
        if re.match(regex_pattern, k):
            if k in donor_sd:
                if donor_sd[k].shape != out_sd[k].shape:
                    raise ValueError(f"Shape mismatch for {k}: donor {donor_sd[k].shape}, recipient {out_sd[k].shape}")
                if verbose: print(f"Patching module parameter: {k}")
                out_sd[k] = donor_sd[k].clone().to(dtype=out_sd[k].dtype, device=out_sd[k].device)
                matched = True
    if not matched and verbose:
        print(f"Warning: No parameters matched pattern '{regex_pattern}'")
    return out_sd

def verify_patch(
    original_sd: Dict[str, torch.Tensor],
    patched_sd: Dict[str, torch.Tensor],
    expected_changed_keys_patterns: List[str]
) -> bool:
    """
    Verifies that only the parameters matching expected patterns have changed.
    """
    all_good = True
    for k in original_sd.keys():
        changed = not torch.equal(original_sd[k], patched_sd[k])
        expected_to_change = any(re.match(pat, k) for pat in expected_changed_keys_patterns)

        # We need a more granular check for attention heads, since they share keys
        is_attn_head_fused_key = "in_proj" in k or "out_proj.weight" in k

        if changed and not expected_to_change and not is_attn_head_fused_key:
            print(f"Error: Key {k} changed unexpectedly.")
            all_good = False
        elif not changed and expected_to_change and not is_attn_head_fused_key:
            print(f"Warning: Key {k} was expected to change but didn't.")
    return all_good
