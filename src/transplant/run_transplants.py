"""
Experiment A — surgical circuit transplant matrix.

Hypothesis: a contaminated (label-noise) run that fails to grok is missing a
specific Fourier circuit that the matched pure run developed.

This script implements a matrix experiment: it takes a list of checkpoints
representing different severities and transplants candidate components (e.g.
attention heads, MLPs, embeddings, LayerNorms) from every donor to every
recipient. It evaluates zero-shot accuracy, and optionally brief finetuned
accuracy, writing out a CSV and heatmap visualizations.

Supports an ablation scaling mode: testing a fraction of a component to see
if rescue is gradual or all-or-nothing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic
    from src.train import evaluate
    from src.transplant.circuits import (
        patch_state_dict,
        patch_state_dict_fractional,
        COMPONENT_PATTERNS
    )
except ImportError:
    from model import ModularArithmeticTransformer  # type: ignore
    from data import DatasetConfig, generate_modular_arithmetic  # type: ignore
    from train import evaluate  # type: ignore
    from circuits import (  # type: ignore
        patch_state_dict,
        patch_state_dict_fractional,
        COMPONENT_PATTERNS
    )

DEFAULT_PATCH_COMPONENTS = [
    "token_embed",
    "pos_embed",
    "attn_all",
    "mlp_all",
    "norm_all",
    "output_head",
]

def load_run(run_dir: Path, step: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Return (state_dict, config) for the given run.
    If step is None, picks the largest checkpoint."""
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_*.pt in {run_dir}")
    chosen: Optional[Path] = None
    if step is not None:
        for p in ckpts:
            if int(re.findall(r"\d+", p.name)[-1]) == step:
                chosen = p
                break
        if chosen is None:
            raise FileNotFoundError(f"no checkpoint_{step}.pt in {run_dir}")
    else:
        chosen = ckpts[-1]

    ckpt = torch.load(chosen, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
        cfg = ckpt.get("config", {})
    else:
        sd = ckpt
        cfg = {}

    res_path = run_dir / "results.json"
    if not cfg and res_path.exists():
        with res_path.open() as f:
            cfg = json.load(f).get("config", {})
    return sd, cfg

def build_model(cfg: dict, device: torch.device) -> ModularArithmeticTransformer:
    return ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
    ).to(device)

def make_loaders(
    cfg: dict, batch_size: int = 512, device: torch.device = torch.device("cpu")
) -> Tuple[DataLoader, DataLoader]:
    """Reconstruct the exact train/test split for the given config."""
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

def freeze_component(model: nn.Module, component: str) -> int:
    """Freeze all params whose state_dict-style name matches `component`."""
    pat = COMPONENT_PATTERNS.get(component, component)
    n_frozen = 0
    for name, p in model.named_parameters():
        if re.match(pat, name):
            p.requires_grad_(False)
            n_frozen += 1
    return n_frozen

def rescue_train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
) -> None:
    """Train trainable params for `steps` steps."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    it = iter(train_loader)
    model.train()
    for s in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    _, test_acc = evaluate(model, test_loader, device)
    return test_acc

def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, out_path: Path):
    pivot = df.pivot(index="donor", columns="recipient", values=value_col)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(pivot.values, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(cax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="left")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Recipient")
    ax.set_ylabel("Donor")
    ax.set_title(title, pad=20)

    for (i, j), val in np.ndenumerate(pivot.values):
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                color="white" if val < 0.5 else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def get_run_name(run_path: Path) -> str:
    # Just use the parent directory name, e.g. noise0.15 or seed_42
    # This might need adapting based on actual directory structure
    # For now, let's use the last two path components to be safe
    return f"{run_path.parent.name}/{run_path.name}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="Paths to run directories to include in the matrix.")
    ap.add_argument("--components", type=str,
                    default=",".join(DEFAULT_PATCH_COMPONENTS),
                    help="Comma-separated components to patch in matrix mode.")
    ap.add_argument("--ablation-component", type=str, default=None,
                    help="If provided, run fractional ablation (10/25/50/75/100%%) on this component instead of matrix.")

    ap.add_argument("--rescue-steps", type=int, default=0,
                    help="Steps of post-patch retraining (0 to disable).")
    ap.add_argument("--rescue-lr", type=float, default=1e-3)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("analysis/transplant_matrix"),
                    help="Where to save results.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for random-basis controls and rescue.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    runs_data = []
    for run_path in args.runs:
        sd, cfg = load_run(run_path)
        name = get_run_name(run_path)
        runs_data.append({
            "name": name,
            "path": run_path,
            "sd": sd,
            "cfg": cfg
        })

    print(f"[info] loaded {len(runs_data)} runs")

    # Pre-build loaders for recipients
    recipient_loaders = {}
    for r in runs_data:
        train_loader, test_loader = make_loaders(r["cfg"], device=device)
        recipient_loaders[r["name"]] = (train_loader, test_loader)

    if args.ablation_component:
        # ABLATION SCALING MODE
        # We assume first run is donor (pure), second run is recipient (contam)
        if len(runs_data) < 2:
            raise ValueError("Ablation mode requires at least 2 runs (donor, recipient)")

        donor = runs_data[0]
        recipient = runs_data[1]

        comp = args.ablation_component
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

        results = []
        train_loader, test_loader = recipient_loaders[recipient["name"]]

        # Baseline
        model = build_model(recipient["cfg"], device)
        model.load_state_dict(recipient["sd"])
        base_acc = evaluate_model(model, test_loader, device)
        results.append({"fraction": 0.0, "test_acc": base_acc, "type": "zero_shot"})

        for frac in fractions:
            patched_sd, meta = patch_state_dict_fractional(
                base_sd=recipient["sd"],
                donor_sd=donor["sd"],
                component=comp,
                fraction=frac,
                n_heads=recipient["cfg"].get("n_heads", 4),
                d_model=recipient["cfg"].get("d_model", 128),
                d_ff=recipient["cfg"].get("d_ff", 512),
                seed=args.seed
            )

            model = build_model(recipient["cfg"], device)
            missing, unexpected = model.load_state_dict(patched_sd, strict=False)
            if unexpected: raise RuntimeError(f"Unexpected keys: {unexpected}")

            # Zero-shot
            zs_acc = evaluate_model(model, test_loader, device)
            results.append({"fraction": frac, "test_acc": zs_acc, "type": "zero_shot"})

            # Rescue
            if args.rescue_steps > 0:
                # Reload fresh model for rescue
                model = build_model(recipient["cfg"], device)
                model.load_state_dict(patched_sd, strict=False)

                freeze_component(model, comp) # This might freeze more than just the fraction, ideally we'd freeze specific parameters

                torch.manual_seed(args.seed)
                rescue_wd = float(recipient["cfg"].get("weight_decay", 1.0))
                rescue_train(model, train_loader, device, args.rescue_steps, args.rescue_lr, rescue_wd)
                rt_acc = evaluate_model(model, test_loader, device)
                results.append({"fraction": frac, "test_acc": rt_acc, "type": "finetuned"})

        df = pd.DataFrame(results)
        df.to_csv(args.output_dir / f"ablation_{comp}.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        for t in df["type"].unique():
            sub = df[df["type"] == t]
            ax.plot(sub["fraction"], sub["test_acc"], marker="o", label=t)
        ax.set_xlabel(f"Fraction of {comp} transplanted")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Ablation scaling: {donor['name']} -> {recipient['name']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"ablation_{comp}.png", dpi=160)
        print(f"[done] ablation sweep saved to {args.output_dir}")

    else:
        # MATRIX MODE
        components = [c.strip() for c in args.components.split(",") if c.strip()]

        results = []
        for comp in components:
            for donor in runs_data:
                for recipient in runs_data:
                    print(f"Transplanting {comp}: {donor['name']} -> {recipient['name']}")

                    patched_sd, meta = patch_state_dict(
                        recipient["sd"], donor["sd"], component=comp
                    )

                    model = build_model(recipient["cfg"], device)
                    missing, unexpected = model.load_state_dict(patched_sd, strict=False)
                    if unexpected: raise RuntimeError(f"Unexpected keys: {unexpected}")

                    train_loader, test_loader = recipient_loaders[recipient["name"]]

                    # Zero-shot
                    zs_acc = evaluate_model(model, test_loader, device)

                    row = {
                        "component": comp,
                        "donor": donor["name"],
                        "recipient": recipient["name"],
                        "zero_shot_acc": zs_acc,
                        "donor_hash": meta["donor_hash"],
                        "recipient_hash": meta["base_hash"],
                        "patched_keys": ",".join(meta["patched_keys"]),
                        "seed": args.seed,
                    }

                    print(f"  [patched] {len(meta['patched_keys'])} keys. Zero-shot acc: {zs_acc:.3f}")

                    # Rescue
                    if args.rescue_steps > 0:
                        model = build_model(recipient["cfg"], device)
                        model.load_state_dict(patched_sd, strict=False)
                        freeze_component(model, comp)

                        torch.manual_seed(args.seed)
                        rescue_wd = float(recipient["cfg"].get("weight_decay", 1.0))
                        rescue_train(model, train_loader, device, args.rescue_steps, args.rescue_lr, rescue_wd)

                        rt_acc = evaluate_model(model, test_loader, device)
                        row["finetuned_acc"] = rt_acc

                    results.append(row)

        df = pd.DataFrame(results)
        df.to_csv(args.output_dir / "donor_recipient_matrix.csv", index=False)

        # Plot heatmaps
        for comp in components:
            sub = df[df["component"] == comp]
            plot_heatmap(
                sub, "zero_shot_acc",
                f"Zero-shot Rescue Accuracy ({comp})",
                args.output_dir / f"heatmap_{comp}_zero_shot.png"
            )

            if args.rescue_steps > 0:
                plot_heatmap(
                    sub, "finetuned_acc",
                    f"Finetuned Rescue Accuracy ({comp}, {args.rescue_steps} steps)",
                    args.output_dir / f"heatmap_{comp}_finetuned.png"
                )

        print(f"[done] Matrix results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
