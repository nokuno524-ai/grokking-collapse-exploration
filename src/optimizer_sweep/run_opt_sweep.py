"""
Sweep harness for optimizer and regularization axes crossed with collapse conditions.
Grid: Optimizer (AdamW, SGD) x Weight Decay x Learning Rate x Condition (pure, collapsed)
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import os
import sys

# Add parent to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from train import TrainConfig, train
    from data import get_all_conditions
except ImportError:
    from src.train import TrainConfig, train
    from src.data import get_all_conditions

DEFAULT_OPTIMIZERS = ["adamw", "sgd"]
DEFAULT_WDS = [0.1, 1.0, 3.0]
DEFAULT_LRS = [1e-4, 1e-3, 5e-3]
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_CONDITIONS = ["pure", "medium_collapse"]

def fmt_opt(opt: str) -> str: return f"opt_{opt}"
def fmt_wd(wd: float) -> str: return f"wd_{wd:g}"
def fmt_lr(lr: float) -> str: return f"lr_{lr:g}"
def fmt_cond(cond: str) -> str: return f"cond_{cond}"

def build_tasks(optimizers: List[str], wds: List[float], lrs: List[float],
                conditions: List[str], seeds: List[int]) -> List[Tuple[str, float, float, str, int]]:
    """Build a list of all parameter combinations."""
    return list(itertools.product(optimizers, wds, lrs, conditions, seeds))

def run_one(optimizer: str, wd: float, lr: float, condition: str, seed: int,
            output_root: str, max_steps: int, dry_run: bool = False):
    """Run a single configuration in the sweep."""
    out_dir = Path(output_root) / fmt_opt(optimizer) / fmt_wd(wd) / fmt_lr(lr) / fmt_cond(condition) / f"seed_{seed}"

    if dry_run:
        print(f"[DRY RUN] Would train: {optimizer} wd={wd} lr={lr} cond={condition} seed={seed}")
        print(f"[DRY RUN] Output dir: {out_dir}")
        return

    all_conds = get_all_conditions()
    if condition not in all_conds:
        raise ValueError(f"Unknown condition: {condition}")

    cond_cfg = all_conds[condition]

    cfg = TrainConfig(
        prime=cond_cfg.prime,
        train_fraction=cond_cfg.train_fraction,
        lr=lr,
        weight_decay=wd,
        optimizer=optimizer,
        collapse_level=cond_cfg.collapse_level,
        collapse_severity=cond_cfg.collapse_severity,
        noise_fraction=cond_cfg.noise_fraction,
        seed=seed,
        condition_name=condition,
        output_dir=str(out_dir.parent), # train appends condition_name, so we go up one level
        max_steps=max_steps,
    )
    # The output structure will be:
    # output_root/opt_X/wd_Y/lr_Z/cond_W/seed_S/cond_W/results.json
    # To fix this, we override condition_name to be the seed dir, and set output_dir to the condition dir
    cfg.output_dir = str(out_dir.parent)
    cfg.condition_name = f"seed_{seed}"

    return train(cfg)

def main():
    ap = argparse.ArgumentParser(description="Optimizer and Regularization Sweep")
    ap.add_argument("--optimizers", type=str,
                    default=",".join(DEFAULT_OPTIMIZERS),
                    help="Comma-separated list of optimizers")
    ap.add_argument("--weight-decays", type=str,
                    default=",".join(f"{x:g}" for x in DEFAULT_WDS),
                    help="Comma-separated list of weight decays")
    ap.add_argument("--learning-rates", type=str,
                    default=",".join(f"{x:g}" for x in DEFAULT_LRS),
                    help="Comma-separated list of learning rates")
    ap.add_argument("--conditions", type=str,
                    default=",".join(DEFAULT_CONDITIONS),
                    help="Comma-separated list of conditions")
    ap.add_argument("--seeds", type=str,
                    default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="Comma-separated list of seeds")
    ap.add_argument("--max-steps", type=int, default=50000,
                    help="Maximum training steps")
    ap.add_argument("--output-dir", type=str, default="results/opt_sweep",
                    help="Directory to save results")
    ap.add_argument("--array-id", type=int, default=None,
                    help="Array ID for parallel execution")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print configs without running")
    ap.add_argument("--print-task-count", action="store_true",
                    help="Print total task count and exit")
    args = ap.parse_args()

    optimizers = [x.strip() for x in args.optimizers.split(",") if x.strip()]
    wds = [float(x) for x in args.weight_decays.split(",") if x.strip()]
    lrs = [float(x) for x in args.learning_rates.split(",") if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    tasks = build_tasks(optimizers, wds, lrs, conditions, seeds)

    if args.print_task_count:
        print(len(tasks))
        return

    if args.array_id is not None:
        if not 0 <= args.array_id < len(tasks):
            raise SystemExit(f"--array-id {args.array_id} out of range [0, {len(tasks)})")
        opt, wd, lr, cond, seed = tasks[args.array_id]
        print(f"[array {args.array_id}/{len(tasks)}] opt={opt} wd={wd} lr={lr} cond={cond} seed={seed}")
        run_one(opt, wd, lr, cond, seed, args.output_dir, args.max_steps, args.dry_run)
        return

    for i, (opt, wd, lr, cond, seed) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] opt={opt} wd={wd} lr={lr} cond={cond} seed={seed}")
        run_one(opt, wd, lr, cond, seed, args.output_dir, args.max_steps, args.dry_run)

if __name__ == "__main__":
    main()
