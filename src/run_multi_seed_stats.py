"""
Multi-seed runner for statistical analysis of grokking cliffs.

Executes a small, fast configuration across multiple seeds for each condition
to generate distributions of trajectories for statistical testing.

HPC invocation (example via Slurm array):
    sbatch --array=0-49 slurm/run_stats.sbatch
    # Where run_stats.sbatch calls:
    # python src/run_multi_seed_stats.py --array-id $SLURM_ARRAY_TASK_ID
"""

import argparse
from pathlib import Path

try:
    from .train import TrainConfig, train
    from .data import get_all_conditions
except ImportError:
    from train import TrainConfig, train
    from data import get_all_conditions


DEFAULT_SEEDS = list(range(42, 52))  # 10 seeds
CONDITION_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]


def build_tasks(seeds):
    conditions = get_all_conditions()
    tasks = []
    for seed in seeds:
        for cond_name in CONDITION_ORDER:
            tasks.append((seed, cond_name, conditions[cond_name]))
    return tasks


def run_one(seed: int, cond_name: str, cond_cfg, output_root: str, max_steps: int, prime: int):
    train_config = TrainConfig(
        prime=prime,  # Use smaller prime for fast smoke tests by default
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=1.0,
        collapse_level=cond_cfg.collapse_level,
        collapse_severity=cond_cfg.collapse_severity,
        seed=seed,
        condition_name=cond_name,
        output_dir=str(Path(output_root) / str(seed) / cond_name),
        max_steps=max_steps,
        eval_every=50,
        log_every=50,
        save_every=max_steps,  # Only save at end
    )
    return train(train_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated list of seeds (default: 42-51)")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--prime", type=int, default=31, help="Prime modulo to use (smaller is faster)")
    parser.add_argument("--output-dir", type=str, default="results/stats_multi_seed")
    parser.add_argument("--array-id", type=int, default=None,
                        help="If set, run only the task with this index (for Slurm arrays)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = build_tasks(seeds)
    print(f"Total tasks: {len(tasks)} ({len(seeds)} seeds × {len(CONDITION_ORDER)} conditions)")

    if args.array_id is not None:
        if not (0 <= args.array_id < len(tasks)):
            raise SystemExit(f"--array-id {args.array_id} out of range [0, {len(tasks)})")
        seed, cond_name, cond_cfg = tasks[args.array_id]
        print(f"[array {args.array_id}] seed={seed} condition={cond_name}")
        run_one(seed, cond_name, cond_cfg, args.output_dir, args.max_steps, args.prime)
        return

    for i, (seed, cond_name, cond_cfg) in enumerate(tasks):
        print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] seed={seed} condition={cond_name}\n{'='*60}")
        run_one(seed, cond_name, cond_cfg, args.output_dir, args.max_steps, args.prime)


if __name__ == "__main__":
    main()
