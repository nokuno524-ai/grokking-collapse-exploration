"""
Multi-seed sweep across the 5 collapse conditions.

Layout:
  results/multi_seed/<seed>/<condition>/results.json

Sequential mode runs all (seed, condition) pairs locally.
Array mode (--array-id N) runs exactly one task; intended for Slurm array jobs.
"""

import argparse
from pathlib import Path

try:
    from .data import get_all_conditions
    from .train import TrainConfig, train
except ImportError:
    from data import get_all_conditions
    from train import TrainConfig, train


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
CONDITION_ORDER = [
    "pure",
    "low_collapse",
    "medium_collapse",
    "high_collapse",
    "severe_collapse",
]


def build_tasks(seeds):
    conditions = get_all_conditions()
    tasks = []
    for seed in seeds:
        for cond_name in CONDITION_ORDER:
            tasks.append((seed, cond_name, conditions[cond_name]))
    return tasks


def run_one(seed: int, cond_name: str, cond_cfg, output_root: str, max_steps: int):
    train_config = TrainConfig(
        prime=cond_cfg.prime,
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=1.0,
        collapse_level=cond_cfg.collapse_level,
        collapse_severity=cond_cfg.collapse_severity,
        seed=seed,
        condition_name=cond_name,
        output_dir=str(Path(output_root) / str(seed)),
        max_steps=max_steps,
    )
    return train(train_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated list of seeds (default: 42,43,44,45,46)",
    )
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/multi_seed")
    parser.add_argument(
        "--array-id",
        type=int,
        default=None,
        help="If set, run only the task with this index (for Slurm arrays)",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = build_tasks(seeds)
    print(f"Total tasks: {
        len(tasks)} ({
        len(seeds)} seeds × {
        len(CONDITION_ORDER)} conditions)")

    if args.array_id is not None:
        if not (0 <= args.array_id < len(tasks)):
            raise SystemExit(
                f"--array-id {args.array_id} out of range [0, {len(tasks)})"
            )
        seed, cond_name, cond_cfg = tasks[args.array_id]
        print(f"[array {args.array_id}] seed={seed} condition={cond_name}")
        run_one(seed, cond_name, cond_cfg, args.output_dir, args.max_steps)
        return

    for i, (seed, cond_name, cond_cfg) in enumerate(tasks):
        print(
            f"\n{'=' * 60}\n[{i + 1}/{len(tasks)}] seed={seed} condition={cond_name}\n{'=' * 60}"
        )
        run_one(seed, cond_name, cond_cfg, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
