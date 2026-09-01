"""
Transplant generalization harness.
Automates circuit transplants between collapse-severity levels and checkpoints.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np

try:
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic
    from src.train import evaluate
    from src.transplant.circuits import (
        patch_state_dict,
        patch_attention_head,
        patch_layer_blocks,
        patch_random_basis,
        shuffle_attention_heads,
        COMPONENT_PATTERNS
    )
    from src.transplant.stats import bootstrap_ci, cohens_d
except ImportError:
    from model import ModularArithmeticTransformer  # type: ignore
    from data import DatasetConfig, generate_modular_arithmetic  # type: ignore
    from train import evaluate  # type: ignore
    from transplant.circuits import (  # type: ignore
        patch_state_dict,
        patch_attention_head,
        patch_layer_blocks,
        patch_random_basis,
        shuffle_attention_heads,
        COMPONENT_PATTERNS
    )
    from transplant.stats import bootstrap_ci, cohens_d  # type: ignore

def load_run(run_dir: Path, step: Optional[int] = None) -> Tuple[Dict, dict]:
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
    sd = torch.load(chosen, map_location="cpu")

    with (run_dir / "results.json").open() as f:
        cfg = json.load(f)["config"]

    return sd, cfg

def build_model(cfg: dict, device: torch.device) -> ModularArithmeticTransformer:
    return ModularArithmeticTransformer(
        prime=int(cfg.get("prime", 59)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        d_ff=int(cfg.get("d_ff", 512)),
        n_layers=int(cfg.get("n_layers", 1)),
    ).to(device)

def get_loaders(cfg: dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    ds_cfg = DatasetConfig(
        prime=int(cfg.get("prime", 59)),
        train_fraction=float(cfg.get("train_fraction", 0.3)),
        noise_fraction=float(cfg.get("noise_fraction", 0.0)),
        collapse_severity=float(cfg.get("collapse_severity", 0.0)),
        collapse_level=float(cfg.get("collapse_level", 0.0)),
        seed=int(cfg.get("seed", 42))
    )
    # unpack ignoring the optional 4th return (collapse_mask) if present
    ret = generate_modular_arithmetic(ds_cfg)
    X_train, y_train, X_test, y_test = ret[:4]

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    test_ds = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)

    return train_loader, test_loader

def run_experiment_matrix(
    pure_runs: List[Path],
    severe_runs: List[Path],
    checkpoints: List[Optional[int]],
    components: List[str],
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Runs transplant operations across seeds, checkpoints, and directions.
    Directions: pure->severe (rescue), severe->pure (sabotage).
    """
    results = []

    for pure_dir, severe_dir in zip(pure_runs, severe_runs):
        for step in checkpoints:
            try:
                pure_sd, pure_cfg = load_run(pure_dir, step)
                severe_sd, severe_cfg = load_run(severe_dir, step)
            except FileNotFoundError as e:
                print(f"Skipping {pure_dir} / {severe_dir} at step {step}: {e}")
                continue

            seed = pure_cfg.get("seed", 42)
            pure_loaders = get_loaders(pure_cfg)
            severe_loaders = get_loaders(severe_cfg)

            model = build_model(pure_cfg, device)

            # Baseline evaluations
            model.load_state_dict(pure_sd, strict=True)
            pure_loss, pure_acc = evaluate(model, pure_loaders[1], device)

            model.load_state_dict(severe_sd, strict=True)
            severe_loss, severe_acc = evaluate(model, severe_loaders[1], device)

            n_heads = model.n_heads
            n_layers = len(model.transformer.layers)

            # Helper to run evaluations and append results
            def run_and_log(sd, c_name, c_type, direction, baseline_acc, baseline_loss, loaders):
                model.load_state_dict(sd, strict=True)
                loss, acc = evaluate(model, loaders[1], device)
                results.append({
                    "direction": direction,
                    "seed": seed,
                    "step": step,
                    "component": c_name,
                    "type": c_type,
                    "acc": acc,
                    "acc_delta": acc - baseline_acc,
                    "loss": loss,
                    "loss_delta": loss - baseline_loss
                })

            # pure -> severe (rescue)
            for comp in components:
                # Basic string components (e.g. token_embed, linear1)
                if comp in COMPONENT_PATTERNS:
                    patched_sd = patch_state_dict(severe_sd, pure_sd, comp)
                    run_and_log(patched_sd, comp, "transplant", "pure->severe", severe_acc, severe_loss, severe_loaders)

                    lesion_sd = patch_random_basis(severe_sd, comp, seed=seed)
                    run_and_log(lesion_sd, comp, "random_basis", "pure->severe", severe_acc, severe_loss, severe_loaders)

            # Extended components: Heads
            for layer_idx in range(n_layers):
                for head_idx in range(n_heads):
                    comp_name = f"layer{layer_idx}_head{head_idx}"

                    patched_sd = patch_attention_head(severe_sd, pure_sd, layer_idx, head_idx, n_heads)
                    run_and_log(patched_sd, comp_name, "transplant", "pure->severe", severe_acc, severe_loss, severe_loaders)

                # Shuffled heads ablation control (per layer)
                comp_name = f"layer{layer_idx}_shuffled_heads"
                lesion_sd = shuffle_attention_heads(severe_sd, layer_idx, n_heads, seed=seed)
                run_and_log(lesion_sd, comp_name, "shuffled_heads", "pure->severe", severe_acc, severe_loss, severe_loaders)

            # Extended components: Layer Blocks (single layers for now)
            for layer_idx in range(n_layers):
                comp_name = f"layer{layer_idx}_block"
                patched_sd = patch_layer_blocks(severe_sd, pure_sd, layer_idx, layer_idx + 1)
                run_and_log(patched_sd, comp_name, "transplant", "pure->severe", severe_acc, severe_loss, severe_loaders)

            # severe -> pure (sabotage)
            for comp in components:
                # Basic string components
                if comp in COMPONENT_PATTERNS:
                    patched_sd = patch_state_dict(pure_sd, severe_sd, comp)
                    run_and_log(patched_sd, comp, "transplant", "severe->pure", pure_acc, pure_loss, pure_loaders)

                    lesion_sd = patch_random_basis(pure_sd, comp, seed=seed)
                    run_and_log(lesion_sd, comp, "random_basis", "severe->pure", pure_acc, pure_loss, pure_loaders)

            # Extended components: Heads
            for layer_idx in range(n_layers):
                for head_idx in range(n_heads):
                    comp_name = f"layer{layer_idx}_head{head_idx}"

                    patched_sd = patch_attention_head(pure_sd, severe_sd, layer_idx, head_idx, n_heads)
                    run_and_log(patched_sd, comp_name, "transplant", "severe->pure", pure_acc, pure_loss, pure_loaders)

                # Shuffled heads ablation control
                comp_name = f"layer{layer_idx}_shuffled_heads"
                lesion_sd = shuffle_attention_heads(pure_sd, layer_idx, n_heads, seed=seed)
                run_and_log(lesion_sd, comp_name, "shuffled_heads", "severe->pure", pure_acc, pure_loss, pure_loaders)

            # Extended components: Layer Blocks
            for layer_idx in range(n_layers):
                comp_name = f"layer{layer_idx}_block"
                patched_sd = patch_layer_blocks(pure_sd, severe_sd, layer_idx, layer_idx + 1)
                run_and_log(patched_sd, comp_name, "transplant", "severe->pure", pure_acc, pure_loss, pure_loaders)
    return results

def format_results_markdown(aggregated: List[Dict[str, Any]]) -> str:
    lines = [
        "| Direction | Checkpoint | Component | Type | Acc Δ (mean ± std) | Acc 95% CI | Acc Cohen's d | Loss Δ (mean ± std) | Loss 95% CI | Loss Cohen's d |",
        "|---|---|---|---|---|---|---|---|---|---|"
    ]
    for row in aggregated:
        acc_ci_str = f"[{row['acc_delta_ci_lower']:.3f}, {row['acc_delta_ci_upper']:.3f}]"
        loss_ci_str = f"[{row['loss_delta_ci_lower']:.3f}, {row['loss_delta_ci_upper']:.3f}]"

        acc_d_str = f"{row['acc_cohens_d']:.3f}" if not np.isnan(row['acc_cohens_d']) else "NaN"
        loss_d_str = f"{row['loss_cohens_d']:.3f}" if not np.isnan(row['loss_cohens_d']) else "NaN"

        lines.append(
            f"| {row['direction']} | {row['step'] or 'last'} | {row['component']} | {row['type']} | "
            f"{row['acc_delta_mean']:.3f} ± {row['acc_delta_std']:.3f} | {acc_ci_str} | {acc_d_str} | "
            f"{row['loss_delta_mean']:.3f} ± {row['loss_delta_std']:.3f} | {loss_ci_str} | {loss_d_str} |"
        )
    return "\n".join(lines)

def aggregate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Group by direction, step, component, type
    grouped = {}
    for r in results:
        key = (r['direction'], r['step'], r['component'], r['type'])
        if key not in grouped:
            grouped[key] = {"acc_deltas": [], "loss_deltas": [], "acc_baseline": [], "acc_patched": [], "loss_baseline": [], "loss_patched": []}
        grouped[key]["acc_deltas"].append(r['acc_delta'])
        grouped[key]["loss_deltas"].append(r['loss_delta'])
        grouped[key]["acc_baseline"].append(r['acc'] - r['acc_delta'])
        grouped[key]["acc_patched"].append(r['acc'])
        grouped[key]["loss_baseline"].append(r['loss'] - r['loss_delta'])
        grouped[key]["loss_patched"].append(r['loss'])

    aggregated = []
    for key, vals in grouped.items():
        direction, step, component, mtype = key
        acc_deltas = vals["acc_deltas"]
        loss_deltas = vals["loss_deltas"]

        acc_mean, acc_lower, acc_upper = bootstrap_ci(acc_deltas)
        loss_mean, loss_lower, loss_upper = bootstrap_ci(loss_deltas)

        acc_d = cohens_d(vals["acc_patched"], vals["acc_baseline"])
        loss_d = cohens_d(vals["loss_patched"], vals["loss_baseline"])

        aggregated.append({
            "direction": direction,
            "step": step,
            "component": component,
            "type": mtype,
            "acc_delta_mean": float(np.mean(acc_deltas)),
            "acc_delta_std": float(np.std(acc_deltas)),
            "acc_delta_ci_lower": acc_lower,
            "acc_delta_ci_upper": acc_upper,
            "acc_cohens_d": acc_d,
            "loss_delta_mean": float(np.mean(loss_deltas)),
            "loss_delta_std": float(np.std(loss_deltas)),
            "loss_delta_ci_lower": loss_lower,
            "loss_delta_ci_upper": loss_upper,
            "loss_cohens_d": loss_d,
            "n_seeds": len(acc_deltas)
        })

    return aggregated

def main():
    parser = argparse.ArgumentParser(description="Run transplant experiments.")
    parser.add_argument("--pure-dirs", type=str, required=True, nargs='+', help="Paths to pure runs (e.g. wd1/noise0/seed_1 ...)")
    parser.add_argument("--severe-dirs", type=str, required=True, nargs='+', help="Paths to severe runs matching the seeds (e.g. wd1/noise0.15/seed_1 ...)")
    parser.add_argument("--checkpoints", type=int, nargs='+', default=[None], help="Specific step(s) to load. Defaults to last checkpoint.")
    parser.add_argument("--components", type=str, nargs='+', default=["token_embed", "self_attn_in_proj", "self_attn_out_proj", "linear1", "linear2", "output_head"])
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/transplant"))

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pure_paths = [Path(p) for p in args.pure_dirs]
    severe_paths = [Path(p) for p in args.severe_dirs]

    if len(pure_paths) != len(severe_paths):
        raise ValueError("Number of pure runs must match number of severe runs.")

    print(f"Running across {len(pure_paths)} seed pairs...")

    raw_results = run_experiment_matrix(
        pure_paths,
        severe_paths,
        args.checkpoints,
        args.components,
        device
    )

    aggregated = aggregate_results(raw_results)

    # Save CSV
    import csv
    csv_path = args.output_dir / "transplant_results.csv"
    if aggregated:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregated[0].keys())
            writer.writeheader()
            writer.writerows(aggregated)
    print(f"Wrote CSV to {csv_path}")

    # Save Markdown
    md_path = args.output_dir / "transplant_results.md"
    md_str = format_results_markdown(aggregated)
    md_path.write_text(md_str)
    print(f"Wrote Markdown to {md_path}")

if __name__ == "__main__":
    main()
