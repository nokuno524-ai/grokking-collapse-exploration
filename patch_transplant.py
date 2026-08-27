import re

with open("src/transplant_rescue.py", "r") as f:
    content = f.read()

# Update COMPONENT_PATTERNS
comp_patterns_search = """COMPONENT_PATTERNS: Dict[str, str] = {
    "token_embed": r"^token_embed\\.",
    "pos_embed": r"^pos_embed\\.",
    "self_attn_in_proj": r"^transformer\\.layers\\.\\d+\\.self_attn\\.in_proj_(weight|bias)$",
    "self_attn_out_proj": r"^transformer\\.layers\\.\\d+\\.self_attn\\.out_proj\\.",
    "linear1": r"^transformer\\.layers\\.\\d+\\.linear1\\.",
    "linear2": r"^transformer\\.layers\\.\\d+\\.linear2\\.",
    "norm1": r"^transformer\\.layers\\.\\d+\\.norm1\\.",
    "norm2": r"^transformer\\.layers\\.\\d+\\.norm2\\.",
    "ln": r"^ln\\.",
    "output_head": r"^output_head\\.",
}"""

comp_patterns_replace = """COMPONENT_PATTERNS: Dict[str, str] = {
    "token_embed": r"^token_embed\\.",
    "pos_embed": r"^pos_embed\\.",
    "self_attn_in_proj": r"^transformer\\.layers\\.\\d+\\.self_attn\\.in_proj_(weight|bias)$",
    "self_attn_out_proj": r"^transformer\\.layers\\.\\d+\\.self_attn\\.out_proj\\.",
    "linear1": r"^transformer\\.layers\\.\\d+\\.linear1\\.",
    "linear2": r"^transformer\\.layers\\.\\d+\\.linear2\\.",
    "norm1": r"^transformer\\.layers\\.\\d+\\.norm1\\.",
    "norm2": r"^transformer\\.layers\\.\\d+\\.norm2\\.",
    "ln": r"^ln\\.",
    "output_head": r"^output_head\\.",
    "layer_L_head_H": r"^transformer\\.layers\\.\\d+\\.self_attn\\.(in_proj_(weight|bias)|out_proj\\.weight)$"
}"""

content = content.replace(comp_patterns_search, comp_patterns_replace)

# Update keys_for
keys_for_search = """def keys_for(component: str, sd: Dict[str, torch.Tensor]) -> List[str]:
    pat = COMPONENT_PATTERNS[component]
    return [k for k in sd.keys() if re.match(pat, k)]"""

keys_for_replace = """def keys_for(component: str, sd: Dict[str, torch.Tensor]) -> List[str]:
    # Special handling for specific head transplant
    head_match = re.match(r"layer_(\\d+)_head_(\\d+)", component)
    if head_match:
        layer_idx = head_match.group(1)
        pat = f"^transformer\\.layers\\.{layer_idx}\\.self_attn\\.(in_proj_(weight|bias)|out_proj\\.weight)$"
        return [k for k in sd.keys() if re.match(pat, k)]

    if component in COMPONENT_PATTERNS:
        pat = COMPONENT_PATTERNS[component]
    else:
        raise ValueError(f"Unknown component pattern: {component}")
    return [k for k in sd.keys() if re.match(pat, k)]"""

content = content.replace(keys_for_search, keys_for_replace)

# Update patch_state_dict
patch_state_dict_search = """def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
    randomize: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    \"\"\"Return a new state_dict = base, with `component` replaced by donor's.
    If randomize=True, replace with a random-orthonormal-basis version of base
    (specificity control); donor is unused in that path.\"\"\"
    out = {k: v.clone() for k, v in base_sd.items()}
    if randomize:
        if rng is None:
            rng = torch.Generator().manual_seed(0)
        for k in keys_for(component, base_sd):
            out[k] = random_basis_swap(base_sd[k], rng)
    else:
        for k in keys_for(component, base_sd):
            if k not in donor_sd:
                continue
            if donor_sd[k].shape != base_sd[k].shape:
                raise ValueError(
                    f"shape mismatch on {k}: donor {tuple(donor_sd[k].shape)} "
                    f"vs base {tuple(base_sd[k].shape)}"
                )
            out[k] = donor_sd[k].clone()
    return out"""

patch_state_dict_replace = """def patch_state_dict(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    component: str,
    randomize: bool = False,
    rng: Optional[torch.Generator] = None,
    n_heads: int = 4, # Used for specific head transplants
) -> Dict[str, torch.Tensor]:
    \"\"\"Return a new state_dict = base, with `component` replaced by donor's.
    If randomize=True, replace with a random-orthonormal-basis version of base
    (specificity control); donor is unused in that path.\"\"\"
    out = {k: v.clone() for k, v in base_sd.items()}

    # Check if this is a specific head transplant
    head_match = re.match(r"layer_(\\d+)_head_(\\d+)", component)
    if head_match:
        layer_idx = int(head_match.group(1))
        head_idx = int(head_match.group(2))

        for k in keys_for(component, base_sd):
            if "in_proj" in k or "out_proj" in k:
                d_model = base_sd[k].shape[-1]
                head_dim = d_model // n_heads
                start_idx = head_idx * head_dim
                end_idx = start_idx + head_dim

                # Clone to avoid mutating original
                mod_tensor = out[k].clone()

                if "in_proj_weight" in k:
                    # in_proj_weight is shape (3 * d_model, d_model)
                    # Q, K, V are concatenated along dim 0
                    for proj_idx in range(3):
                        qkv_start = proj_idx * d_model + start_idx
                        qkv_end = proj_idx * d_model + end_idx
                        if randomize:
                            if rng is None: rng = torch.Generator().manual_seed(0)
                            mod_tensor[qkv_start:qkv_end, :] = random_basis_swap(base_sd[k][qkv_start:qkv_end, :], rng)
                        else:
                            mod_tensor[qkv_start:qkv_end, :] = donor_sd[k][qkv_start:qkv_end, :]
                elif "in_proj_bias" in k:
                    # in_proj_bias is shape (3 * d_model)
                    for proj_idx in range(3):
                        qkv_start = proj_idx * d_model + start_idx
                        qkv_end = proj_idx * d_model + end_idx
                        if randomize:
                            if rng is None: rng = torch.Generator().manual_seed(0)
                            mod_tensor[qkv_start:qkv_end] = random_basis_swap(base_sd[k][qkv_start:qkv_end], rng)
                        else:
                            mod_tensor[qkv_start:qkv_end] = donor_sd[k][qkv_start:qkv_end]
                elif "out_proj.weight" in k:
                    # out_proj is shape (d_model, d_model)
                    # Attention head outputs are concatenated along dim 1
                    if randomize:
                        if rng is None: rng = torch.Generator().manual_seed(0)
                        mod_tensor[:, start_idx:end_idx] = random_basis_swap(base_sd[k][:, start_idx:end_idx], rng)
                    else:
                        mod_tensor[:, start_idx:end_idx] = donor_sd[k][:, start_idx:end_idx]

                out[k] = mod_tensor
        return out

    if randomize:
        if rng is None:
            rng = torch.Generator().manual_seed(0)
        for k in keys_for(component, base_sd):
            out[k] = random_basis_swap(base_sd[k], rng)
    else:
        for k in keys_for(component, base_sd):
            if k not in donor_sd:
                continue
            if donor_sd[k].shape != base_sd[k].shape:
                raise ValueError(
                    f"shape mismatch on {k}: donor {tuple(donor_sd[k].shape)} "
                    f"vs base {tuple(base_sd[k].shape)}"
                )
            out[k] = donor_sd[k].clone()
    return out"""

content = content.replace(patch_state_dict_search, patch_state_dict_replace)

# Update run_one_variant
run_one_variant_search = """def run_one_variant(
    name: str,
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Optional[Dict[str, torch.Tensor]],
    component: Optional[str],
    cfg_for_loaders: dict,
    cfg_for_model: dict,
    device: torch.device,
    randomize: bool = False,
    rescue_steps: int = 0,
    rescue_lr: float = 1e-3,
    rescue_wd: float = 1.0,
    rescue_seed: int = 0,
    rng: Optional[torch.Generator] = None,
) -> VariantResult:
    if component is None:
        # baseline / swap_all path: state dict is taken as-is from `donor_sd`
        sd = {k: v.clone() for k, v in (donor_sd or base_sd).items()}
    else:
        sd = patch_state_dict(base_sd, donor_sd or {}, component,
                              randomize=randomize, rng=rng)"""

run_one_variant_replace = """def run_one_variant(
    name: str,
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Optional[Dict[str, torch.Tensor]],
    component: Optional[str],
    cfg_for_loaders: dict,
    cfg_for_model: dict,
    device: torch.device,
    randomize: bool = False,
    rescue_steps: int = 0,
    rescue_lr: float = 1e-3,
    rescue_wd: float = 1.0,
    rescue_seed: int = 0,
    rng: Optional[torch.Generator] = None,
) -> VariantResult:
    if component is None:
        # baseline / swap_all path: state dict is taken as-is from `donor_sd`
        sd = {k: v.clone() for k, v in (donor_sd or base_sd).items()}
    else:
        n_heads = int(cfg_for_model.get("n_heads", 4))
        sd = patch_state_dict(base_sd, donor_sd or {}, component,
                              randomize=randomize, rng=rng, n_heads=n_heads)"""

content = content.replace(run_one_variant_search, run_one_variant_replace)

# update argparse for matching component
main_comp_search = """    for c in components:
        if c not in COMPONENT_PATTERNS:
            raise ValueError(f"unknown component {c!r}; valid: {list(COMPONENT_PATTERNS)}")"""

main_comp_replace = """    for c in components:
        if c not in COMPONENT_PATTERNS and not re.match(r"layer_\\d+_head_\\d+", c):
            raise ValueError(f"unknown component {c!r}; valid: {list(COMPONENT_PATTERNS)} or layer_L_head_H")"""

content = content.replace(main_comp_search, main_comp_replace)

with open("src/transplant_rescue.py", "w") as f:
    f.write(content)
