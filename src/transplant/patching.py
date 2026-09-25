"""
Core state-dict patching and slicing utilities.
"""
import re
from typing import Dict, List, Optional
import torch

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

# The 6 components we report in the main table — picked because each is a single
# semantically meaningful block. token_embed and output_head touch the Fourier
# basis directly; attn / FFN are the obvious circuit-carrying matrices.
DEFAULT_PATCH_COMPONENTS = [
    "token_embed",
    "self_attn_in_proj",
    "self_attn_out_proj",
    "linear1",
    "linear2",
    "output_head",
]

def keys_for(component: str, sd: Dict[str, torch.Tensor]) -> List[str]:
    """Return all keys in `sd` matching the regex for `component`."""
    pat = COMPONENT_PATTERNS[component]
    return [k for k in sd.keys() if re.match(pat, k)]

def random_basis_swap(weight: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Return a tensor with the same shape and spectrum as `weight` but a
    random orthonormal basis. Specificity control: keeps ||W||_F and the
    singular values, randomizes which directions are which.

    For 1-D bias vectors, returns a copy with a random *permutation* of the
    same values (preserves L2 norm and entry distribution)."""
    orig_device = weight.device
    orig_dtype = weight.dtype
    w = weight.detach().to(torch.float32).clone()
    if w.ndim == 1:
        idx = torch.randperm(w.numel(), generator=rng)
        out = w[idx]
    elif w.ndim != 2:
        # higher-dim parameters: flatten the last dims and apply 2-D variant
        orig_shape = w.shape
        w2 = w.reshape(w.shape[0], -1)
        out = random_basis_swap(w2, rng)
        out = out.reshape(orig_shape)
    else:
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        # Random orthonormal U' and Vh' of correct shapes.
        Ur, _ = torch.linalg.qr(torch.randn(U.shape, generator=rng))
        Vr, _ = torch.linalg.qr(torch.randn(Vh.T.shape, generator=rng))
        Vhr = Vr.T
        out = Ur @ torch.diag(S) @ Vhr
    return out.to(dtype=orig_dtype, device=orig_device)

def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
    randomize: bool = False,
    rng: Optional[torch.Generator] = None,
    key_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """Return a new state_dict = base, with `component` replaced by donor's.
    If randomize=True, replace with a random-orthonormal-basis version of base
    (specificity control); donor is unused in that path.
    If key_mapping is provided, it maps donor keys to base keys."""
    out = {k: v.clone() for k, v in base_sd.items()}
    if randomize:
        if rng is None:
            rng = torch.Generator().manual_seed(0)
        for k in keys_for(component, base_sd):
            out[k] = random_basis_swap(base_sd[k], rng)
    else:
        for base_k in keys_for(component, base_sd):
            # Find the corresponding donor key, defaulting to same key
            donor_k = base_k
            if key_mapping:
                # Reverse mapping: we need to find which donor key maps to this base key
                for dk, bk in key_mapping.items():
                    if bk == base_k:
                        donor_k = dk
                        break

            if donor_k not in donor_sd:
                raise KeyError(f"Donor key {donor_k} not found for base key {base_k}")

            if donor_sd[donor_k].shape != base_sd[base_k].shape:
                raise ValueError(
                    f"shape mismatch on {base_k}: donor {tuple(donor_sd[donor_k].shape)} "
                    f"vs base {tuple(base_sd[base_k].shape)}"
                )
            out[base_k] = donor_sd[donor_k].clone().to(dtype=base_sd[base_k].dtype, device=base_sd[base_k].device)
    return out

def patch_head_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    d_head: int,
    num_heads: int,
) -> Dict[str, torch.Tensor]:
    """
    Patch a single attention head from `donor_sd` into `base_sd`.
    Handles fused QKV projection weights/biases and output projection.
    """
    out = {k: v.clone() for k, v in base_sd.items()}

    in_proj_weight_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
    in_proj_bias_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"
    out_proj_weight_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"

    d_model = num_heads * d_head

    # Patch in_proj (Q, K, V are stacked)
    if in_proj_weight_key in base_sd and in_proj_weight_key in donor_sd:
        # in_proj_weight is shape (3 * d_model, d_model)
        for i in range(3): # For Q, K, V
            start = i * d_model + head_idx * d_head
            end = start + d_head
            out[in_proj_weight_key][start:end, :] = donor_sd[in_proj_weight_key][start:end, :].clone().to(dtype=base_sd[in_proj_weight_key].dtype, device=base_sd[in_proj_weight_key].device)

    if in_proj_bias_key in base_sd and in_proj_bias_key in donor_sd:
        # in_proj_bias is shape (3 * d_model)
        for i in range(3):
            start = i * d_model + head_idx * d_head
            end = start + d_head
            out[in_proj_bias_key][start:end] = donor_sd[in_proj_bias_key][start:end].clone().to(dtype=base_sd[in_proj_bias_key].dtype, device=base_sd[in_proj_bias_key].device)

    # Patch out_proj
    if out_proj_weight_key in base_sd and out_proj_weight_key in donor_sd:
        # out_proj.weight is shape (d_model, d_model)
        start = head_idx * d_head
        end = start + d_head
        out[out_proj_weight_key][:, start:end] = donor_sd[out_proj_weight_key][:, start:end].clone().to(dtype=base_sd[out_proj_weight_key].dtype, device=base_sd[out_proj_weight_key].device)

    return out
