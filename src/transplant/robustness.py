import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import copy
import re

from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic
from src.train import evaluate
from src.transplant.head_transplant import transplant_head

@dataclass
class RobustnessConfig:
    pure_ckpt_path: Path
    contam_ckpt_path: Path
    eval_seeds: List[int]
    recompute_layernorm: bool

def _force_layernorm_stats_hook(module: nn.LayerNorm, input: tuple, output: torch.Tensor, target_mean: torch.Tensor, target_var: torch.Tensor):
    """
    Hook to force the LayerNorm output to have specific mean and variance.
    This effectively turns off layer norm recomputation for the patched model
    so it uses the unpatched model's normalization scale.
    """
    # x_normalized = (x - mean) / sqrt(var + eps)
    # y = x_normalized * weight + bias

    # We want to use target_mean and target_var instead of the batch's actual mean/var.
    x = input[0]
    # Re-normalize using the provided target stats
    x_norm = (x - target_mean) / torch.sqrt(target_var + module.eps)
    y = x_norm * module.weight + module.bias
    return y

def evaluate_transplant_head(
    base_model: nn.Module,
    donor_model: nn.Module,
    layer_idx: int,
    head_idx: int,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: DatasetConfig,
    recompute_layernorm: bool = True
) -> float:
    """
    Evaluate the model with a specific head transplanted.
    Returns test accuracy.
    """
    patched_sd = transplant_head(
        base_model.state_dict(),
        donor_model.state_dict(),
        layer_idx,
        head_idx,
        d_model=config.d_model if hasattr(config, 'd_model') else 128,
        n_heads=config.n_heads if hasattr(config, 'n_heads') else 4
    )

    # We don't have config.d_model in DatasetConfig right now, default to 128
    model_config = {"prime": config.prime, "d_model": 128, "n_heads": 4}
    patched_model = ModularArithmeticTransformer(**model_config).to(device)
    patched_model.load_state_dict(patched_sd)

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
            return args, kwargs

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
        return correct / total

def compute_head_importance(
    pure_model: nn.Module,
    contam_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: DatasetConfig,
    recompute_layernorm: bool
) -> np.ndarray:
    """
    Compute importance score for each head by transplanting from pure to contam
    and measuring the increase in accuracy.
    Returns array of shape (n_layers, n_heads).
    """
    n_layers = 1 # hardcoded for this architecture
    n_heads = 4

    base_loss, base_acc = evaluate(contam_model, test_loader, device)

    importance = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            acc = evaluate_transplant_head(
                contam_model, pure_model, l, h,
                test_loader, device, config, recompute_layernorm
            )
            importance[l, h] = acc - base_acc

    return importance

def check_robustness(
    pure_dir: Path,
    contam_dir: Path,
    device: torch.device,
    seed_variations: List[int] = [42, 100, 200]
) -> Dict[str, float]:
    """
    Run robustness checks for head attribution.
    Returns dict of correlations.
    """
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
    }
