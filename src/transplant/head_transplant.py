"""
Head-wise and layer-wise circuit transplantation tool.

Performs targeted surgical swaps of specific attention heads, layers, or MLP blocks
between two matched checkpoints (e.g., pure grokked vs collapsed).

It also supports random ablation controls (replacing a component with a random orthogonal
basis) to distinguish mechanistic importance from simple capacity reduction.
"""

import argparse
import copy
import json
import logging
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic
    from src.train import compute_fourier_concentration, evaluate
    from src.transplant.transplant_rescue import random_basis_swap, get_fourier_basis, load_run
except ImportError:
    from ..model import ModularArithmeticTransformer
    from ..data import DatasetConfig, generate_modular_arithmetic
    from ..train import compute_fourier_concentration, evaluate
    from .transplant_rescue import random_basis_swap, get_fourier_basis, load_run

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class TransplantResult:
    base_run: str
    donor_run: str
    variant: str
    test_loss: float
    test_acc: float
    train_loss: float
    train_acc: float
    weight_norm: float


def split_qkv(weight: torch.Tensor, d_model: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused in_proj_weight/bias (3*d_model, ...) into Q, K, V."""
    assert weight.shape[0] == 3 * d_model
    return weight[:d_model], weight[d_model:2*d_model], weight[2*d_model:]


def splice_head_weight(
    base_w: torch.Tensor,
    donor_w: torch.Tensor,
    head_idx: int,
    n_heads: int,
    d_model: int,
    is_qkv: bool = False,
    is_out: bool = False,
    ablate: bool = False,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Splice a single head's weight from donor to base, or ablate it."""
    head_dim = d_model // n_heads
    start = head_idx * head_dim
    end = start + head_dim

    out = base_w.clone()

    def get_donor_slice(b: torch.Tensor, d: torch.Tensor, s: int, e: int, axis: int) -> torch.Tensor:
        if ablate:
            sl = b.clone()
            if axis == 0:
                target = sl[s:e]
            else:
                target = sl[:, s:e]
            # Ablate with a random orthogonal basis
            return random_basis_swap(target, rng)
        else:
            if axis == 0:
                return d[s:e]
            else:
                return d[:, s:e]

    if is_qkv:
        # q, k, v are stacked on dim 0
        bq, bk, bv = split_qkv(base_w, d_model)
        dq, dk, dv = split_qkv(donor_w, d_model) if not ablate else (None, None, None)

        bq[start:end] = get_donor_slice(bq, dq, start, end, axis=0)
        bk[start:end] = get_donor_slice(bk, dk, start, end, axis=0)
        bv[start:end] = get_donor_slice(bv, dv, start, end, axis=0)

        out = torch.cat([bq, bk, bv], dim=0)

    elif is_out:
        # out_proj groups heads on dim 1 (in_features) for weight, but not for bias?
        # wait, out_proj weight is (d_model, d_model). The input is from heads concatenated.
        # So dim 1 is head dim.
        if out.ndim == 2:
            out[:, start:end] = get_donor_slice(base_w, donor_w, start, end, axis=1)
        # bias is just (d_model), but it's applied after combining heads, so you can't really
        # attribute parts of the bias to specific heads cleanly. Usually we just leave it or swap all.
        # We will not swap out_proj bias for a single head.

    return out


def patch_head(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    n_heads: int,
    d_model: int,
    ablate: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Patch a specific head in a specific layer."""
    prefix = f"transformer.layers.{layer_idx}.self_attn"
    out = {k: v.clone() for k, v in base_sd.items()}

    # in_proj_weight
    in_w_key = f"{prefix}.in_proj_weight"
    if in_w_key in base_sd:
        out[in_w_key] = splice_head_weight(
            base_sd[in_w_key],
            donor_sd.get(in_w_key) if not ablate else None,
            head_idx, n_heads, d_model, is_qkv=True, ablate=ablate, rng=rng
        )

    # in_proj_bias
    in_b_key = f"{prefix}.in_proj_bias"
    if in_b_key in base_sd:
        out[in_b_key] = splice_head_weight(
            base_sd[in_b_key],
            donor_sd.get(in_b_key) if not ablate else None,
            head_idx, n_heads, d_model, is_qkv=True, ablate=ablate, rng=rng
        )

    # out_proj.weight
    out_w_key = f"{prefix}.out_proj.weight"
    if out_w_key in base_sd:
        out[out_w_key] = splice_head_weight(
            base_sd[out_w_key],
            donor_sd.get(out_w_key) if not ablate else None,
            head_idx, n_heads, d_model, is_out=True, ablate=ablate, rng=rng
        )

    return out


def patch_mlp(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    ablate: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Patch the MLP block (linear1, linear2) of a specific layer."""
    prefix = f"transformer.layers.{layer_idx}"
    out = {k: v.clone() for k, v in base_sd.items()}

    for k in base_sd:
        if k.startswith(f"{prefix}.linear1") or k.startswith(f"{prefix}.linear2"):
            if ablate:
                out[k] = random_basis_swap(base_sd[k], rng)
            else:
                out[k] = donor_sd[k].clone()
    return out


def patch_layer(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    ablate: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Patch an entire transformer layer (attention + MLP + norms)."""
    prefix = f"transformer.layers.{layer_idx}."
    out = {k: v.clone() for k, v in base_sd.items()}

    for k in base_sd:
        if k.startswith(prefix):
            if ablate:
                out[k] = random_basis_swap(base_sd[k], rng)
            else:
                out[k] = donor_sd[k].clone()
    return out



    main()

def make_loaders(
    cfg: dict, batch_size: int = 512, device: torch.device = torch.device("cpu")
) -> Tuple[DataLoader, DataLoader]:
    """Reconstruct the exact train/test split for the given config (matched seed)."""
    dc = DatasetConfig(
        prime=int(cfg.get("prime", 59)),
        train_fraction=float(cfg.get("train_fraction", 0.3)),
        collapse_level=float(cfg.get("collapse_level", 0.0)),
        collapse_severity=float(cfg.get("collapse_severity", 0.5)),
        noise_fraction=float(cfg.get("noise_fraction", 0.0)),
        seed=int(cfg.get("seed", 42)),
    )
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(dc)
    train_ds = TensorDataset(train_in, train_tgt)
    test_ds = TensorDataset(test_in, test_tgt)
    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 42)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates the model on train and test loaders, computing accuracy, loss, and weight norm."""
    train_loss, train_acc = evaluate(model, train_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    wn = float(sum(p.detach().norm().item() ** 2 for p in model.parameters()) ** 0.5)
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "weight_norm": wn,
    }

def run_transplant(
    base_run: Path,
    donor_run: Optional[Path],
    variant: str,
    cfg: dict,
    base_sd: Dict[str, torch.Tensor],
    patched_sd: Dict[str, torch.Tensor],
    device: torch.device,
) -> TransplantResult:
    """Evaluate a transplanted state dict and return a TransplantResult."""
    model = ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
    ).to(device)

    missing, unexpected = model.load_state_dict(patched_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading patched sd: {unexpected}")

    train_loader, test_loader = make_loaders(cfg, device=device)
    metrics = evaluate_model(model, train_loader, test_loader, device)

    return TransplantResult(
        base_run=str(base_run.name) if base_run else "None",
        donor_run=str(donor_run.name) if donor_run else "None",
        variant=variant,
        test_loss=metrics["test_loss"],
        test_acc=metrics["test_acc"],
        train_loss=metrics["train_loss"],
        train_acc=metrics["train_acc"],
        weight_norm=metrics["weight_norm"],
    )


def parse_hybrid_components(arg: str) -> List[str]:
    """Parse a comma-separated list of components like head_0, mlp_0, layer_0"""
    return [x.strip() for x in arg.split(",") if x.strip()]

def apply_hybrid_patch(
    base_sd: Dict[str, torch.Tensor],
    donor_sd: Dict[str, torch.Tensor],
    components: List[str],
    n_heads: int,
    d_model: int,
    ablate: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Apply a combination of patches."""
    current_sd = {k: v.clone() for k, v in base_sd.items()}

    for comp in components:
        if comp.startswith("head_"):
            # Format: head_{layer}_{head} or head_{head} (assuming layer 0)
            parts = comp.split("_")
            if len(parts) == 2:
                layer_idx, head_idx = 0, int(parts[1])
            else:
                layer_idx, head_idx = int(parts[1]), int(parts[2])
            current_sd = patch_head(current_sd, donor_sd, layer_idx, head_idx, n_heads, d_model, ablate, rng)
        elif comp.startswith("mlp_"):
            parts = comp.split("_")
            layer_idx = int(parts[1]) if len(parts) > 1 else 0
            current_sd = patch_mlp(current_sd, donor_sd, layer_idx, ablate, rng)
        elif comp.startswith("layer_"):
            parts = comp.split("_")
            layer_idx = int(parts[1]) if len(parts) > 1 else 0
            current_sd = patch_layer(current_sd, donor_sd, layer_idx, ablate, rng)
        else:
            raise ValueError(f"Unknown hybrid component format: {comp}")

    return current_sd


def run_grid(
    base_dirs: List[Path],
    donor_dirs: List[Path],
    output_dir: Path,
    hybrid_components: str = "",
    ablate: bool = False,
    seed: int = 42,
) -> None:
    """Executes a systematic transplant grid evaluating baseline, head-wise, layer-wise, and MLP components."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    rng = torch.Generator().manual_seed(seed)
    results = []

    for base_dir in base_dirs:
        base_sd, base_cfg = load_run(base_dir)
        n_heads = int(base_cfg.get("n_heads", 4))
        d_model = int(base_cfg.get("d_model", 128))
        n_layers = int(base_cfg.get("n_layers", 1))

        # Determine donor dirs (if not ablating, we need matched donors)
        current_donor_dirs = donor_dirs if not ablate else [None]

        for donor_dir in current_donor_dirs:
            if donor_dir is not None:
                donor_sd, donor_cfg = load_run(donor_dir)
                if int(base_cfg.get("seed", -1)) != int(donor_cfg.get("seed", -2)):
                    logging.warning(f"Seed mismatch between {base_dir} and {donor_dir}")
            else:
                donor_sd = None
                donor_cfg = None

            # 1. Baseline eval
            res = run_transplant(base_dir, donor_dir, "baseline", base_cfg, base_sd, base_sd, device)
            results.append(res)

            # 2. Layer-wise eval
            for layer_idx in range(n_layers):
                patched = patch_layer(base_sd, donor_sd, layer_idx, ablate, rng)
                res = run_transplant(base_dir, donor_dir, f"layer_{layer_idx}", base_cfg, base_sd, patched, device)
                results.append(res)

                # 3. MLP eval
                patched = patch_mlp(base_sd, donor_sd, layer_idx, ablate, rng)
                res = run_transplant(base_dir, donor_dir, f"mlp_{layer_idx}", base_cfg, base_sd, patched, device)
                results.append(res)

                # 4. Head-wise eval
                for head_idx in range(n_heads):
                    patched = patch_head(base_sd, donor_sd, layer_idx, head_idx, n_heads, d_model, ablate, rng)
                    res = run_transplant(base_dir, donor_dir, f"head_{layer_idx}_{head_idx}", base_cfg, base_sd, patched, device)
                    results.append(res)

            # 5. Hybrid eval
            if hybrid_components:
                comps = parse_hybrid_components(hybrid_components)
                patched = apply_hybrid_patch(base_sd, donor_sd, comps, n_heads, d_model, ablate, rng)
                res = run_transplant(base_dir, donor_dir, f"hybrid_{hybrid_components}", base_cfg, base_sd, patched, device)
                results.append(res)

    # Output to CSV and Markdown
    import csv
    csv_path = output_dir / "transplant_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        if results:
            writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    md_path = output_dir / "transplant_summary.md"
    with open(md_path, "w") as f:
        f.write("# Transplant Results\n\n")
        f.write("| base_run | donor_run | variant | train_acc | test_acc | test_loss |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r.base_run} | {r.donor_run} | {r.variant} | {r.train_acc:.3f} | {r.test_acc:.3f} | {r.test_loss:.3f} |\n")

    logging.info(f"Saved results to {csv_path} and {md_path}")

    parser = argparse.ArgumentParser(description="Head-wise and layer-wise circuit transplantation.")
    parser.add_argument("--base-dirs", nargs="+", type=Path, required=True, help="Base runs to graft into.")
    parser.add_argument("--donor-dirs", nargs="*", type=Path, default=[], help="Donor runs to copy from.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results.")
    parser.add_argument("--hybrid-components", type=str, default="", help="Comma-separated components to patch together (e.g. head_0_0,mlp_0)")
    parser.add_argument("--ablate", action="store_true", help="Perform random ablation instead of transplantation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ablation.")
    args = parser.parse_args()

    if not args.ablate and not args.donor_dirs:
        parser.error("--donor-dirs is required unless --ablate is used.")

    run_grid(
        args.base_dirs,
        args.donor_dirs,
        args.output_dir,
        args.hybrid_components,
        args.ablate,
        args.seed
    )


def main():
    """Main CLI entry point for circuit transplantation experiments."""
    parser = argparse.ArgumentParser(description="Head-wise and layer-wise circuit transplantation.")
    parser.add_argument("--base-dirs", nargs="+", type=Path, required=True, help="Base runs to graft into.")
    parser.add_argument("--donor-dirs", nargs="*", type=Path, default=[], help="Donor runs to copy from.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results.")
    parser.add_argument("--hybrid-components", type=str, default="", help="Comma-separated components to patch together (e.g. head_0_0,mlp_0)")
    parser.add_argument("--ablate", action="store_true", help="Perform random ablation instead of transplantation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ablation.")
    args = parser.parse_args()

    if not args.ablate and not args.donor_dirs:
        parser.error("--donor-dirs is required unless --ablate is used.")

    run_grid(
        args.base_dirs,
        args.donor_dirs,
        args.output_dir,
        args.hybrid_components,
        args.ablate,
        args.seed
    )

if __name__ == "__main__":
    main()
