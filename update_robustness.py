import re
import os

with open("src/transplant/robustness.py", "r") as f:
    content = f.read()

# Replace the evaluate_transplant_head layernorm logic
old_logic = """    hooks = []
    if not recompute_layernorm:
        # We need to compute the base model's layernorm stats for the first batch
        # For simplicity, we just attach a hook to compute it per batch on the base model,
        # and then apply it to the patched model.
        # However, a simpler implementation is to just hook the patched model's layernorm
        # forward pass to use the base model's output stats.
        pass # Will implement hook logic if needed, but standard is recompute_layernorm=True
        # To avoid complex hook management in this simplified robust driver, we will skip the
        # actual hook for now unless explicitly requested, but keep the flag for the API.

    _, acc = evaluate(patched_model, test_loader, device)

    for h in hooks:
        h.remove()

    return acc"""

new_logic = """    hooks = []
    if recompute_layernorm:
        _, acc = evaluate(patched_model, test_loader, device)
        return acc
    else:
        # Layernorm recomputation off: Use original base model's stats
        base_ln_inputs = []
        def base_ln_hook(module, args, kwargs):
            x = args[0]
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            base_ln_inputs.append((mean, var))
            return args

        h_base = base_model.ln.register_forward_pre_hook(base_ln_hook, with_kwargs=True)

        def patched_ln_forward_hook(module, args, output):
            x = args[0]
            if base_ln_inputs:
                target_mean, target_var = base_ln_inputs.pop(0)
            else:
                target_mean, target_var = x.mean(dim=-1, keepdim=True), x.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x - target_mean) / torch.sqrt(target_var + module.eps)
            return x_norm * module.weight + module.bias

        h_patched = patched_model.ln.register_forward_hook(patched_ln_forward_hook)

        patched_model.eval()
        base_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                _ = base_model(x)
                logits = patched_model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        h_base.remove()
        h_patched.remove()
        return correct / total"""

content = content.replace(old_logic, new_logic)

# Re-write check_robustness cleanly
old_check_full = """def check_robustness(
    pure_dir: Path,
    contam_dir: Path,
    device: torch.device,
    seed_variations: List[int] = [42, 100, 200]
) -> Dict[str, float]:
    \"\"\"
    Run robustness checks for head attribution.
    Returns dict of correlations.
    \"\"\"
    # Load last checkpt
    pure_ckpts = sorted(pure_dir.glob("checkpoint_*.pt"), key=lambda p: int(re.findall(r"\\d+", p.name)[-1]))
    contam_ckpts = sorted(contam_dir.glob("checkpoint_*.pt"), key=lambda p: int(re.findall(r"\\d+", p.name)[-1]))

    pure_ckpt = torch.load(pure_ckpts[-1], map_location="cpu")
    contam_ckpt = torch.load(contam_ckpts[-1], map_location="cpu")

    config = DatasetConfig(**pure_ckpt.get("config", {}))

    pure_model = ModularArithmeticTransformer(prime=config.prime).to(device)
    pure_model.load_state_dict(pure_ckpt["model_state"])

    contam_model = ModularArithmeticTransformer(prime=config.prime).to(device)
    contam_model.load_state_dict(contam_ckpt["model_state"])

    importances = []

    for seed in seed_variations:
        # Generate new eval batch with different seed
        cfg = copy.deepcopy(config)
        cfg.seed = seed
        _, _, test_in, test_tgt = generate_modular_arithmetic(cfg)

        test_ds = torch.utils.data.TensorDataset(test_in, test_tgt)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)

        # Base settings
        imp = compute_head_importance(pure_model, contam_model, test_loader, device, cfg, True)
        importances.append(imp.flatten())

        # Layernorm variation
        imp_no_ln = compute_head_importance(pure_model, contam_model, test_loader, device, cfg, False)
        importances.append(imp_no_ln.flatten())

    # Compute pairwise spearman rank correlations between all variations
    correlations = []
    n_vars = len(importances)
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            # If arrays are constant or all zero, spearmanr returns nan
            if np.all(importances[i] == importances[i][0]) or np.all(importances[j] == importances[j][0]):
                correlations.append(0.0)
            else:
                rho, _ = scipy.stats.spearmanr(importances[i], importances[j])
                # Handle nan
                if np.isnan(rho):
                    rho = 1.0 if np.allclose(importances[i], importances[j]) else 0.0
                correlations.append(rho)

    return {
        "mean_correlation": float(np.mean(correlations)),
        "min_correlation": float(np.min(correlations))
    }"""

new_check_full = """def check_robustness(
    pure_dir: Path,
    contam_dir: Path,
    device: torch.device,
    seed_variations: List[int] = [42, 100, 200]
) -> Dict[str, float]:
    \"\"\"
    Run robustness checks for head attribution.
    Returns dict of correlations.
    \"\"\"
    pure_ckpts = sorted(pure_dir.glob("checkpoint_*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    contam_ckpts = sorted(contam_dir.glob("checkpoint_*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[-1]))

    importances = []

    # We want to vary the checkpoint pair selection.
    # Sample 3 pairs: early, middle, and late
    ckpt_indices = [
        max(0, len(pure_ckpts) // 4),
        max(0, len(pure_ckpts) // 2),
        -1
    ]

    for idx in ckpt_indices:
        if not pure_ckpts or not contam_ckpts:
            continue
        if idx >= len(pure_ckpts): idx = -1
        if idx >= len(contam_ckpts): idx = -1

        pure_ckpt = torch.load(pure_ckpts[idx], map_location="cpu")
        contam_ckpt = torch.load(contam_ckpts[idx], map_location="cpu")

        config = DatasetConfig(**pure_ckpt.get("config", {}))

        pure_model = ModularArithmeticTransformer(prime=config.prime).to(device)
        pure_model.load_state_dict(pure_ckpt["model_state"])

        contam_model = ModularArithmeticTransformer(prime=config.prime).to(device)
        contam_model.load_state_dict(contam_ckpt["model_state"])

        for seed in seed_variations:
            # Generate new eval batch with different seed
            cfg = copy.deepcopy(config)
            cfg.seed = seed
            _, _, test_in, test_tgt = generate_modular_arithmetic(cfg)

            test_ds = torch.utils.data.TensorDataset(test_in, test_tgt)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)

            # Base settings
            imp = compute_head_importance(pure_model, contam_model, test_loader, device, cfg, True)
            importances.append(imp.flatten())

            # Layernorm variation
            imp_no_ln = compute_head_importance(pure_model, contam_model, test_loader, device, cfg, False)
            importances.append(imp_no_ln.flatten())

    if not importances:
        return {"mean_correlation": 0.0, "min_correlation": 0.0}

    # Compute pairwise spearman rank correlations between all variations
    correlations = []
    n_vars = len(importances)
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            # If arrays are constant or all zero, spearmanr returns nan
            if np.all(importances[i] == importances[i][0]) or np.all(importances[j] == importances[j][0]):
                correlations.append(0.0)
            else:
                rho, _ = scipy.stats.spearmanr(importances[i], importances[j])
                # Handle nan
                if np.isnan(rho):
                    rho = 1.0 if np.allclose(importances[i], importances[j]) else 0.0
                correlations.append(rho)

    return {
        "mean_correlation": float(np.mean(correlations)) if correlations else 0.0,
        "min_correlation": float(np.min(correlations)) if correlations else 0.0
    }"""

# A simple regex matching to ensure we don't trip over \d vs \\d
content = re.sub(
    r'def check_robustness\(.*?\n    }\n',
    new_check_full + '\n',
    content,
    flags=re.DOTALL
)

with open("src/transplant/robustness.py", "w") as f:
    f.write(content)
