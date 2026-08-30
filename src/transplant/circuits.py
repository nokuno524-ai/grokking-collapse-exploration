import re
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any

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
    # Additional aggregate components
    "attn_all": r"^transformer\.layers\.\d+\.self_attn\.",
    "mlp_all": r"^transformer\.layers\.\d+\.linear[12]\.",
    "embed_all": r"^(token|pos)_embed\.",
    "norm_all": r"^transformer\.layers\.\d+\.norm[12]\.|^ln\.",
}

def clean_key(key: str) -> str:
    """Strip DDP/compilation prefixes from state_dict keys."""
    if key.startswith("module."):
        key = key[7:]
    if key.startswith("_orig_mod."):
        key = key[10:]
    return key

def get_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    """Compute a SHA256 hash of a state_dict's parameters to identify checkpoints."""
    hasher = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        # Hash the shape and flat values
        hasher.update(k.encode())
        hasher.update(str(v.shape).encode())
        if v.numel() > 0:
            # We hash the sum and a few elements to be reasonably fast but unique
            # For exact uniqueness, we could hash the raw bytes but that's slow.
            # Using norm + mean + first few elements is generally enough.
            hasher.update(str(v.norm().item()).encode())
            hasher.update(str(v.mean().item()).encode())
    return hasher.hexdigest()

def keys_for(component: str, sd: Dict[str, torch.Tensor]) -> List[str]:
    """Get all keys in sd that match the component pattern."""
    pat = COMPONENT_PATTERNS.get(component, component) # fallback to treating component as regex
    return [k for k in sd.keys() if re.match(pat, clean_key(k))]

def random_basis_swap(weight: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Return a tensor with the same shape and spectrum but a random orthonormal basis."""
    w = weight.detach().to(torch.float32).clone()
    if w.ndim == 1:
        idx = torch.randperm(w.numel(), generator=rng)
        return w[idx]
    if w.ndim != 2:
        orig_shape = w.shape
        w2 = w.reshape(w.shape[0], -1)
        out = random_basis_swap(w2, rng)
        return out.reshape(orig_shape)
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    Ur, _ = torch.linalg.qr(torch.randn(U.shape, generator=rng, device=w.device))
    Vr, _ = torch.linalg.qr(torch.randn(Vh.T.shape, generator=rng, device=w.device))
    Vhr = Vr.T
    return Ur @ torch.diag(S) @ Vhr

def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
    randomize: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Return a new state_dict with `component` replaced by donor's.
    Returns (new_sd, metadata_dict).
    """
    out = {}
    base_cleaned_to_orig = {clean_key(k): k for k in base_sd.keys()}
    donor_cleaned_to_orig = {clean_key(k): k for k in donor_sd.keys()}

    # Init output with cloned base
    for k, v in base_sd.items():
        out[k] = v.clone()

    matched_keys = []

    if randomize:
        if rng is None:
            rng = torch.Generator(device=next(iter(base_sd.values())).device).manual_seed(0)
        target_keys = keys_for(component, {clean_key(k): v for k, v in base_sd.items()})
        for ck in target_keys:
            orig_k = base_cleaned_to_orig[ck]
            out[orig_k] = random_basis_swap(base_sd[orig_k], rng)
            matched_keys.append(orig_k)
    else:
        target_keys = keys_for(component, {clean_key(k): v for k, v in base_sd.items()})
        for ck in target_keys:
            orig_k = base_cleaned_to_orig[ck]

            if ck not in donor_cleaned_to_orig:
                continue

            donor_orig_k = donor_cleaned_to_orig[ck]
            donor_tensor = donor_sd[donor_orig_k]
            base_tensor = base_sd[orig_k]

            if donor_tensor.shape != base_tensor.shape:
                raise ValueError(
                    f"shape mismatch on {ck}: donor {tuple(donor_tensor.shape)} "
                    f"vs base {tuple(base_tensor.shape)}"
                )

            # ensure devices match
            donor_cloned = donor_tensor.clone().to(base_tensor.device)
            out[orig_k] = donor_cloned
            matched_keys.append(orig_k)

    metadata = {
        "component": component,
        "patched_keys": matched_keys,
        "base_hash": get_hash(base_sd),
        "donor_hash": get_hash(donor_sd) if not randomize else None,
        "randomized": randomize
    }

    return out, metadata

def patch_state_dict_fractional(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
    fraction: float,
    n_heads: int = 4,
    d_model: int = 128,
    d_ff: int = 512,
    seed: int = 42,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Transplant a specific fraction of a component (e.g., 50% of attention heads or 25% of MLP neurons).
    Currently supports:
      - 'attn' (fraction of heads)
      - 'mlp' (fraction of hidden neurons)
    """
    out = {}
    for k, v in base_sd.items():
        out[k] = v.clone()

    rng = torch.Generator().manual_seed(seed)

    base_cleaned_to_orig = {clean_key(k): k for k in base_sd.keys()}
    donor_cleaned_to_orig = {clean_key(k): k for k in donor_sd.keys()}

    metadata = {
        "component": component,
        "fraction": fraction,
        "patched_keys": [],
        "base_hash": get_hash(base_sd),
        "donor_hash": get_hash(donor_sd),
        "seed": seed
    }

    if component == "attn":
        n_heads_to_patch = int(n_heads * fraction)
        if n_heads_to_patch == 0:
            return out, metadata

        # randomly select heads to patch
        head_indices = torch.randperm(n_heads, generator=rng)[:n_heads_to_patch].tolist()
        head_dim = d_model // n_heads

        # In multi-head attention, in_proj is usually (3 * d_model, d_model)
        # out_proj is (d_model, d_model)

        # Patch in_proj
        for k in keys_for("self_attn_in_proj", {clean_key(k): v for k, v in base_sd.items()}):
            ck = clean_key(k)
            orig_k = base_cleaned_to_orig[ck]
            donor_orig_k = donor_cleaned_to_orig.get(ck)
            if not donor_orig_k: continue

            base_t = base_sd[orig_k].clone()
            donor_t = donor_sd[donor_orig_k].to(base_t.device)

            if base_t.ndim == 2: # weight
                # in_proj_weight is [3 * d_model, d_model]
                # Q, K, V are each [d_model, d_model]
                # For each, heads are chunked [n_heads, head_dim]
                for h in head_indices:
                    for i in range(3): # Q, K, V
                        start_idx = i * d_model + h * head_dim
                        end_idx = i * d_model + (h + 1) * head_dim
                        base_t[start_idx:end_idx, :] = donor_t[start_idx:end_idx, :]
            elif base_t.ndim == 1: # bias
                for h in head_indices:
                    for i in range(3):
                        start_idx = i * d_model + h * head_dim
                        end_idx = i * d_model + (h + 1) * head_dim
                        base_t[start_idx:end_idx] = donor_t[start_idx:end_idx]
            out[orig_k] = base_t
            metadata["patched_keys"].append(f"{orig_k} (heads: {head_indices})")

        # Patch out_proj
        for k in keys_for("self_attn_out_proj", {clean_key(k): v for k, v in base_sd.items()}):
            ck = clean_key(k)
            orig_k = base_cleaned_to_orig[ck]
            donor_orig_k = donor_cleaned_to_orig.get(ck)
            if not donor_orig_k: continue

            base_t = base_sd[orig_k].clone()
            donor_t = donor_sd[donor_orig_k].to(base_t.device)

            if "weight" in ck:
                # out_proj_weight is [d_model, d_model], but it projects FROM [d_model] which is the concat of head outputs.
                # So the input dimension (dim=1) corresponds to heads.
                for h in head_indices:
                    start_idx = h * head_dim
                    end_idx = (h + 1) * head_dim
                    base_t[:, start_idx:end_idx] = donor_t[:, start_idx:end_idx]
                out[orig_k] = base_t
                metadata["patched_keys"].append(f"{orig_k} (heads: {head_indices})")
            # out_proj bias is added after merging all heads, so we typically don't patch it fractionally
            # or we patch it fully if fraction==1.0, but for now we skip bias fractional patching.

    elif component == "mlp":
        n_neurons_to_patch = int(d_ff * fraction)
        if n_neurons_to_patch == 0:
            return out, metadata

        neuron_indices = torch.randperm(d_ff, generator=rng)[:n_neurons_to_patch].tolist()

        # linear1 is [d_ff, d_model]
        for k in keys_for("linear1", {clean_key(k): v for k, v in base_sd.items()}):
            ck = clean_key(k)
            orig_k = base_cleaned_to_orig[ck]
            donor_orig_k = donor_cleaned_to_orig.get(ck)
            if not donor_orig_k: continue

            base_t = base_sd[orig_k].clone()
            donor_t = donor_sd[donor_orig_k].to(base_t.device)

            if base_t.ndim == 2: # weight
                base_t[neuron_indices, :] = donor_t[neuron_indices, :]
            elif base_t.ndim == 1: # bias
                base_t[neuron_indices] = donor_t[neuron_indices]
            out[orig_k] = base_t
            metadata["patched_keys"].append(f"{orig_k} (neurons: {len(neuron_indices)})")

        # linear2 is [d_model, d_ff]
        for k in keys_for("linear2", {clean_key(k): v for k, v in base_sd.items()}):
            ck = clean_key(k)
            orig_k = base_cleaned_to_orig[ck]
            donor_orig_k = donor_cleaned_to_orig.get(ck)
            if not donor_orig_k: continue

            base_t = base_sd[orig_k].clone()
            donor_t = donor_sd[donor_orig_k].to(base_t.device)

            if base_t.ndim == 2: # weight
                # maps FROM d_ff TO d_model, so input dim is dim 1
                base_t[:, neuron_indices] = donor_t[:, neuron_indices]
            # bias is [d_model], doesn't correspond to neurons, don't patch fractionally.
            out[orig_k] = base_t
            metadata["patched_keys"].append(f"{orig_k} (neurons: {len(neuron_indices)})")
    else:
        raise ValueError(f"Fractional patch not supported for {component}")

    return out, metadata
