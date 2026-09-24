"""
Systematic circuit transplant sweep.

This script runs transplant experiments (both module-level and head-level)
across all pairs of conditions (e.g. pure vs low/medium/severe collapse) to
isolate where generalization fails.

Usage:
  python src/transplant/sweep.py --base-dir results/exp_c_grid/wd1 --dry-run
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn

try:
    from .transplant_rescue import load_run, COMPONENT_PATTERNS, make_loaders, evaluate_model, build_model
    from .patching import patch_state_dict, patch_head_state_dict
except ImportError:
    from src.transplant.transplant_rescue import load_run, COMPONENT_PATTERNS, make_loaders, evaluate_model, build_model
    from src.transplant.patching import patch_state_dict, patch_head_state_dict


# The modules to swap for module-level sweeping
MODULES_TO_SWAP = [
    "token_embed",
    "pos_embed",
    "self_attn_in_proj",
    "self_attn_out_proj",
    "linear1",
    "linear2",
    "norm1",
    "norm2",
    "ln",
    "output_head",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Systematic transplant sweep")
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="Base directory containing condition subdirectories")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan the swaps without executing them")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/transplant_sweep"),
                        help="Where to save the localization matrix")
    return parser.parse_args()

def discover_runs(base_dir: Path) -> Dict[str, List[Path]]:
    """Discover all runs grouped by condition in the base directory."""
    runs_by_condition = {}
    if not base_dir.exists():
        logging.warning(f"Base dir {base_dir} does not exist.")
        return runs_by_condition

    for condition_dir in base_dir.iterdir():
        if not condition_dir.is_dir():
            continue
        condition = condition_dir.name
        runs = []
        for seed_dir in condition_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                # verify it has a checkpoint
                if any(seed_dir.glob("checkpoint_*.pt")):
                    runs.append(seed_dir)
        if runs:
            runs_by_condition[condition] = runs

    return runs_by_condition

def pair_runs(runs_by_condition: Dict[str, List[Path]]) -> List[Tuple[str, str, Path, Path]]:
    """
    Pair donor (pure) and recipient (collapsed) runs by seed.
    Returns: list of (donor_cond, recipient_cond, donor_path, recipient_path)
    """
    pairs = []
    conditions = sorted(runs_by_condition.keys())

    for donor_cond in conditions:
        for recipient_cond in conditions:
            # We pair all combinations, including self-pairs (as a sanity check)
            donor_runs = runs_by_condition[donor_cond]
            recipient_runs = runs_by_condition[recipient_cond]

            # Match by seed
            for donor_path in donor_runs:
                donor_seed = donor_path.name
                for recipient_path in recipient_runs:
                    if recipient_path.name == donor_seed:
                        pairs.append((donor_cond, recipient_cond, donor_path, recipient_path))
                        break # Only one match per seed

    return pairs

def run_transplants_for_pair(
    donor_path: Path,
    recipient_path: Path,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Run module-level and head-level zero-shot transplants for a pair of runs.
    """
    results = []

    try:
        donor_sd, donor_cfg = load_run(donor_path)
        recipient_sd, recipient_cfg = load_run(recipient_path)
    except Exception as e:
        logging.error(f"Failed to load runs: {e}")
        return results

    train_loader, test_loader = make_loaders(recipient_cfg, device=device)

    # 1. Baseline Recipient
    model = build_model(recipient_cfg, device)
    model.load_state_dict(recipient_sd)
    baseline_metrics = evaluate_model(model, train_loader, test_loader, device)
    baseline_acc = baseline_metrics["test_acc"]

    results.append({
        "module": "baseline_recipient",
        "test_acc": baseline_acc,
        "acc_change": 0.0,
    })

    # 2. Module-level swaps
    for component in MODULES_TO_SWAP:
        patched_sd = patch_state_dict(recipient_sd, donor_sd, component, COMPONENT_PATTERNS)
        model = build_model(recipient_cfg, device)
        model.load_state_dict(patched_sd, strict=False) # strict=False to allow unused keys in some architectures

        metrics = evaluate_model(model, train_loader, test_loader, device)
        acc = metrics["test_acc"]

        results.append({
            "module": component,
            "test_acc": acc,
            "acc_change": acc - baseline_acc,
        })

    # 3. Fine-grained Head-level swaps
    n_layers = int(recipient_cfg.get("n_layers", 1))
    n_heads = int(recipient_cfg.get("n_heads", 4))
    d_model = int(recipient_cfg.get("d_model", 128))

    for l in range(n_layers):
        for h in range(n_heads):
            patched_sd = patch_head_state_dict(recipient_sd, donor_sd, layer_idx=l, head_idx=h, d_model=d_model, n_heads=n_heads)
            model = build_model(recipient_cfg, device)
            model.load_state_dict(patched_sd, strict=False)

            metrics = evaluate_model(model, train_loader, test_loader, device)
            acc = metrics["test_acc"]

            module_name = f"L{l}_H{h}"
            results.append({
                "module": module_name,
                "test_acc": acc,
                "acc_change": acc - baseline_acc,
            })

    return results

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    runs_by_condition = discover_runs(args.base_dir)
    logging.info(f"Found conditions: {list(runs_by_condition.keys())}")

    pairs = pair_runs(runs_by_condition)
    logging.info(f"Formed {len(pairs)} pairs matching by seed")

    if args.dry_run:
        logging.info("DRY RUN: Planning swaps")
        for d_cond, r_cond, d_path, r_path in pairs:
            print(f"  Donor: {d_cond} ({d_path.name}) -> Recipient: {r_cond} ({r_path.name})")
            print(f"    Swaps: Modules {MODULES_TO_SWAP} + Heads (L_H)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    all_results = []

    for d_cond, r_cond, d_path, r_path in pairs:
        logging.info(f"Running transplants for Donor: {d_cond} -> Recipient: {r_cond} ({d_path.name})")
        pair_results = run_transplants_for_pair(d_path, r_path, device)

        for res in pair_results:
            all_results.append({
                "donor_condition": d_cond,
                "recipient_condition": r_cond,
                "seed": d_path.name,
                "module": res["module"],
                "test_acc": res["test_acc"],
                "acc_change": res["acc_change"],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_results(all_results, args.output_dir)

def export_results(all_results: List[Dict[str, Any]], output_dir: Path):
    """
    Export the localization matrix to JSON, CSV, and generate a heatmap.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. JSON
    json_path = output_dir / "localization_matrix.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Saved JSON to {json_path}")

    # 2. CSV
    csv_path = output_dir / "localization_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["donor_condition", "recipient_condition", "seed", "module", "test_acc", "acc_change"])
        writer.writeheader()
        writer.writerows(all_results)
    logging.info(f"Saved CSV to {csv_path}")

    # 3. Heatmaps
    # Group by (donor_cond, recipient_cond) and average across seeds for each module
    avg_results = {}
    for r in all_results:
        key = (r["donor_condition"], r["recipient_condition"], r["module"])
        if key not in avg_results:
            avg_results[key] = []
        avg_results[key].append(r["acc_change"])

    for k in avg_results:
        avg_results[k] = sum(avg_results[k]) / len(avg_results[k])

    conditions = sorted(list(set([r["donor_condition"] for r in all_results])))
    modules = sorted(list(set([r["module"] for r in all_results if r["module"] != "baseline_recipient"])))

    # One heatmap per module
    for module in modules:
        matrix = np.zeros((len(conditions), len(conditions)))
        for i, d_cond in enumerate(conditions):
            for j, r_cond in enumerate(conditions):
                matrix[i, j] = avg_results.get((d_cond, r_cond, module), 0.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(matrix, cmap="RdBu", vmin=-1.0, vmax=1.0)
        fig.colorbar(cax)

        ax.set_xticks(range(len(conditions)))
        ax.set_yticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha="left")
        ax.set_yticklabels(conditions)
        ax.set_xlabel("Recipient Condition")
        ax.set_ylabel("Donor Condition")
        ax.set_title(f"Zero-shot Acc Change - Module: {module}")

        for i in range(len(conditions)):
            for j in range(len(conditions)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if abs(val) > 0.5 else "black")

        heatmap_path = output_dir / f"heatmap_{module}.png"
        fig.tight_layout()
        fig.savefig(heatmap_path)
        plt.close(fig)

    logging.info(f"Saved heatmaps to {output_dir}")

if __name__ == "__main__":
    main()
