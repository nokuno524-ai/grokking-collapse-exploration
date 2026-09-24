import re
from typing import Dict, List, Optional
import torch

def keys_for(component: str, sd: Dict[str, torch.Tensor], patterns: Dict[str, str]) -> List[str]:
    """Return keys in `sd` that match the regex pattern for `component`."""
    pat = patterns[component]
    return [k for k in sd.keys() if re.match(pat, k)]

def random_basis_swap(weight: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Return a tensor with the same shape and spectrum as `weight` but a
    random orthonormal basis. Specificity control: keeps ||W||_F and the
    singular values, randomizes which directions are which.

    For 1-D bias vectors, returns a copy with a random *permutation* of the
    same values (preserves L2 norm and entry distribution)."""
    w = weight.detach().to(torch.float32).clone()
    if w.ndim == 1:
        idx = torch.randperm(w.numel(), generator=rng)
        return w[idx]
    if w.ndim != 2:
        # higher-dim parameters: flatten the last dims and apply 2-D variant
        orig_shape = w.shape
        w2 = w.reshape(w.shape[0], -1)
        out = random_basis_swap(w2, rng)
        return out.reshape(orig_shape)
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    # Random orthonormal U' and Vh' of correct shapes.
    Ur, _ = torch.linalg.qr(torch.randn(U.shape, generator=rng))
    Vr, _ = torch.linalg.qr(torch.randn(Vh.T.shape, generator=rng))
    Vhr = Vr.T
    return Ur @ torch.diag(S) @ Vhr

def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
    patterns: Dict[str, str],
    randomize: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Return a new state_dict = base, with `component` replaced by donor's.
    If randomize=True, replace with a random-orthonormal-basis version of base
    (specificity control); donor is unused in that path."""
    out = {k: v.clone() for k, v in base_sd.items()}
    if randomize:
        if rng is None:
            rng = torch.Generator().manual_seed(0)
        for k in keys_for(component, base_sd, patterns):
            out[k] = random_basis_swap(base_sd[k], rng)
    else:
        for k in keys_for(component, base_sd, patterns):
            if k not in donor_sd:
                continue
            if donor_sd[k].shape != base_sd[k].shape:
                raise ValueError(
                    f"shape mismatch on {k}: donor {tuple(donor_sd[k].shape)} "
                    f"vs base {tuple(base_sd[k].shape)}"
                )
            # Explicitly cast to recipient dtype and device to fix transplant bug
            out[k] = donor_sd[k].clone().to(dtype=base_sd[k].dtype, device=base_sd[k].device)
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
    Patch a single attention head from donor_sd into base_sd.
    Handles slicing of the fused QKV `in_proj_weight`/`bias` matrices and `out_proj`.
    """
    out = {k: v.clone() for k, v in base_sd.items()}
    d_head = d_model // n_heads

    in_proj_w_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
    in_proj_b_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"
    out_proj_w_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"

    # QKV are fused in dim 0: [3 * d_model, d_model]
    for key in [in_proj_w_key, in_proj_b_key]:
        if key in base_sd and key in donor_sd:
            base_t = out[key]
            donor_t = donor_sd[key].to(dtype=base_t.dtype, device=base_t.device)
            for i in range(3): # Q, K, V
                start = i * d_model + head_idx * d_head
                end = start + d_head
                base_t[start:end] = donor_t[start:end]

    # out_proj is [d_model, d_model]
    if out_proj_w_key in base_sd and out_proj_w_key in donor_sd:
        base_t = out[out_proj_w_key]
        donor_t = donor_sd[out_proj_w_key].to(dtype=base_t.dtype, device=base_t.device)
        start = head_idx * d_head
        end = start + d_head
        # For out_proj, heads are concatenated along the input dimension (dim 1)
        base_t[:, start:end] = donor_t[:, start:end]

    return out
