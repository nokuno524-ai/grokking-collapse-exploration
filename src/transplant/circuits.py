"""
Circuit transplanting tools.
Provides functionality for surgical weight transplants between checkpoints,
including key remapping for DDP or torch.compile wrappers.
"""
from typing import Dict, List, Optional
import re
import torch
import torch.nn as nn

# Mapping from a short component name to the regex of state_dict keys it covers.
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

# Expand to support specific layers
for i in range(10): # support up to 10 layers for flexibility
    COMPONENT_PATTERNS[f"layer{i}"] = rf"^transformer\.layers\.{i}\."
    COMPONENT_PATTERNS[f"layer{i}_attn"] = rf"^transformer\.layers\.{i}\.self_attn\."
    COMPONENT_PATTERNS[f"layer{i}_mlp"] = rf"^transformer\.layers\.{i}\.(linear1|linear2)\."

def strip_prefixes(key: str) -> str:
    """Strips DDP (module.) and torch.compile (_orig_mod.) prefixes."""
    k = key
    if k.startswith("module."):
        k = k[7:]
    if k.startswith("_orig_mod."):
        k = k[10:]
    return k

def keys_for(component: str, sd: Dict[str, torch.Tensor]) -> List[str]:
    """Returns keys matching the component pattern, handling prefixes."""
    if component not in COMPONENT_PATTERNS:
        raise ValueError(f"Unknown component {component!r}")

    pat = COMPONENT_PATTERNS[component]
    matching_keys = []

    for k in sd.keys():
        clean_k = strip_prefixes(k)
        if re.match(pat, clean_k):
            matching_keys.append(k)

    return matching_keys

def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
) -> Dict[str, torch.Tensor]:
    """
    Returns a new state_dict = base, with `component` replaced by donor's.
    Enforces shape, dtype, and device match.
    """
    out = {k: v.clone() for k, v in base_sd.items()}

    base_keys = keys_for(component, base_sd)

    if not base_keys:
        return out

    for base_k in base_keys:
        # Find corresponding key in donor
        clean_base_k = strip_prefixes(base_k)

        # We look for a donor key that strips down to the same clean key
        donor_match_k = None
        for dk in donor_sd.keys():
            if strip_prefixes(dk) == clean_base_k:
                donor_match_k = dk
                break

        if donor_match_k is None:
            raise KeyError(f"Donor missing key corresponding to {clean_base_k}")

        d_val = donor_sd[donor_match_k]
        b_val = base_sd[base_k]

        if d_val.shape != b_val.shape:
            raise ValueError(f"Shape mismatch for {clean_base_k}: donor {d_val.shape}, base {b_val.shape}")

        out[base_k] = d_val.to(dtype=b_val.dtype, device=b_val.device).clone()

    return out

def patch_attention_head(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    n_heads: int
) -> Dict[str, torch.Tensor]:
    """
    Returns a new state_dict with a specific attention head from the donor
    transplanted into the base state_dict.

    This patches both in_proj (Q, K, V) and out_proj for the specified head.
    """
    out = {k: v.clone() for k, v in base_sd.items()}

    # 1. Find the keys for in_proj and out_proj of this layer
    in_proj_w_key, in_proj_b_key = None, None
    out_proj_w_key = None # We don't patch out_proj bias as it's not per-head

    for k in base_sd.keys():
        clean_k = strip_prefixes(k)
        if clean_k == f"transformer.layers.{layer_idx}.self_attn.in_proj_weight":
            in_proj_w_key = k
        elif clean_k == f"transformer.layers.{layer_idx}.self_attn.in_proj_bias":
            in_proj_b_key = k
        elif clean_k == f"transformer.layers.{layer_idx}.self_attn.out_proj.weight":
            out_proj_w_key = k

    if not all([in_proj_w_key, in_proj_b_key, out_proj_w_key]):
        raise KeyError(f"Could not find attention weights for layer {layer_idx}")

    # Helper to find corresponding donor key
    def get_donor_key(base_key: str) -> str:
        clean_base = strip_prefixes(base_key)
        for dk in donor_sd.keys():
            if strip_prefixes(dk) == clean_base:
                return dk
        raise KeyError(f"Donor missing key for {clean_base}")

    d_in_proj_w_key = get_donor_key(in_proj_w_key)
    d_in_proj_b_key = get_donor_key(in_proj_b_key)
    d_out_proj_w_key = get_donor_key(out_proj_w_key)

    d_model = out[in_proj_w_key].shape[1]
    head_dim = d_model // n_heads

    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim

    # Patch Q, K, V (in_proj)
    for i in range(3): # Q, K, V are stacked in in_proj
        offset = i * d_model

        # Patch weight
        out[in_proj_w_key][offset + start_idx : offset + end_idx, :] = \
            donor_sd[d_in_proj_w_key][offset + start_idx : offset + end_idx, :].to(
                dtype=out[in_proj_w_key].dtype, device=out[in_proj_w_key].device
            )

        # Patch bias
        out[in_proj_b_key][offset + start_idx : offset + end_idx] = \
            donor_sd[d_in_proj_b_key][offset + start_idx : offset + end_idx].to(
                dtype=out[in_proj_b_key].dtype, device=out[in_proj_b_key].device
            )

    # Patch O (out_proj) - Note: out_proj weight is (d_model, d_model) where columns correspond to heads
    out[out_proj_w_key][:, start_idx : end_idx] = \
        donor_sd[d_out_proj_w_key][:, start_idx : end_idx].to(
            dtype=out[out_proj_w_key].dtype, device=out[out_proj_w_key].device
        )

    return out

def patch_layer_blocks(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    start_layer: int,
    end_layer: int
) -> Dict[str, torch.Tensor]:
    """
    Transplants a block of layers [start_layer, end_layer) from donor to base.
    """
    out = {k: v.clone() for k, v in base_sd.items()}

    for i in range(start_layer, end_layer):
        out = patch_state_dict(out, donor_sd, f"layer{i}")

    return out

def random_basis_swap(weight: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Return a tensor with the same shape and spectrum as `weight` but a
    random orthonormal basis. Specificity control: keeps ||W||_F and the
    singular values, randomizes which directions are which.

    For 1-D bias vectors, returns a copy with a random *permutation* of the
    same values (preserves L2 norm and entry distribution)."""
    w = weight.detach().to(torch.float32).clone()
    if w.ndim == 1:
        idx = torch.randperm(w.numel(), generator=rng)
        return w[idx].to(dtype=weight.dtype)
    if w.ndim != 2:
        # higher-dim parameters: flatten the last dims and apply 2-D variant
        orig_shape = w.shape
        w2 = w.reshape(w.shape[0], -1)
        out = random_basis_swap(w2, rng)
        return out.reshape(orig_shape).to(dtype=weight.dtype)
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    # Random orthonormal U' and Vh' of correct shapes.
    Ur, _ = torch.linalg.qr(torch.randn(U.shape, generator=rng))
    Vr, _ = torch.linalg.qr(torch.randn(Vh.T.shape, generator=rng))
    # reconstruct with original singular values
    w_rand = (Ur * S) @ Vr.T
    return w_rand.to(dtype=weight.dtype)

def patch_random_basis(
    base_sd: Dict[str, torch.Tensor],
    component: str,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Returns a new state_dict with `component` replaced by a random orthonormal
    basis version of the base weights. Acts as an ablation control.
    """
    out = {k: v.clone() for k, v in base_sd.items()}
    rng = torch.Generator().manual_seed(seed)

    base_keys = keys_for(component, base_sd)
    for k in base_keys:
        out[k] = random_basis_swap(out[k], rng).to(device=out[k].device)

    return out

def shuffle_attention_heads(
    base_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    n_heads: int,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Shuffles the attention heads within a layer. Ablation control for head positioning.
    """
    out = {k: v.clone() for k, v in base_sd.items()}
    rng = torch.Generator().manual_seed(seed)

    in_proj_w_key, in_proj_b_key, out_proj_w_key = None, None, None
    for k in base_sd.keys():
        clean_k = strip_prefixes(k)
        if clean_k == f"transformer.layers.{layer_idx}.self_attn.in_proj_weight":
            in_proj_w_key = k
        elif clean_k == f"transformer.layers.{layer_idx}.self_attn.in_proj_bias":
            in_proj_b_key = k
        elif clean_k == f"transformer.layers.{layer_idx}.self_attn.out_proj.weight":
            out_proj_w_key = k

    if not all([in_proj_w_key, in_proj_b_key, out_proj_w_key]):
        raise KeyError(f"Could not find attention weights for layer {layer_idx}")

    d_model = out[in_proj_w_key].shape[1]
    head_dim = d_model // n_heads

    # Generate permutation of head indices
    perm = torch.randperm(n_heads, generator=rng)

    new_in_w = torch.zeros_like(out[in_proj_w_key])
    new_in_b = torch.zeros_like(out[in_proj_b_key])
    new_out_w = torch.zeros_like(out[out_proj_w_key])

    for i in range(n_heads):
        src_idx = i
        tgt_idx = perm[i].item()

        src_start = src_idx * head_dim
        src_end = (src_idx + 1) * head_dim

        tgt_start = tgt_idx * head_dim
        tgt_end = (tgt_idx + 1) * head_dim

        for p in range(3):
            offset = p * d_model
            new_in_w[offset + tgt_start : offset + tgt_end, :] = out[in_proj_w_key][offset + src_start : offset + src_end, :]
            new_in_b[offset + tgt_start : offset + tgt_end] = out[in_proj_b_key][offset + src_start : offset + src_end]

        new_out_w[:, tgt_start : tgt_end] = out[out_proj_w_key][:, src_start : src_end]

    out[in_proj_w_key] = new_in_w
    out[in_proj_b_key] = new_in_b
    out[out_proj_w_key] = new_out_w

    return out
