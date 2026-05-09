"""
Deconfounded 2D grid: collapse_level × collapse_severity, 5 seeds each.

Levels:    [0.0, 0.05, 0.15, 0.30]
Severities: [0.3, 0.6, 0.9]
Seeds:     [42, 43, 44, 45, 46]
=> 4 × 3 × 5 = 60 runs.

Layout:
  results/grid/level<L>_sev<S>/seed_<seed>/results.json
"""

import argparse
from pathlib import Path

try:
    from .train import TrainConfig, train
except ImportError:
    from train import TrainConfig, train


DEFAULT_LEVELS = [0.0, 0.05, 0.15, 0.30]
DEFAULT_SEVERITIES = [0.3, 0.6, 0.9]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def fmt_pair(level: float, severity: float) -> str:
    return f"level{level:g}_sev{severity:g}"


def build_tasks(levels, severities, seeds):
    tasks = []
    for level in levels:
        for severity in severities:
            for seed in seeds:
                tasks.append((level, severity, seed))
    return tasks


def run_one(level: float, severity: float, seed: int, output_root: str, max_steps: int):
    pair_dir = fmt_pair(level, severity)
    train_config = TrainConfig(
        prime=59,
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=1.0,
        collapse_level=level,
        collapse_severity=severity,
        seed=seed,
        condition_name=f"seed_{seed}",
        output_dir=str(Path(output_root) / pair_dir),
        max_steps=max_steps,
    )
    return train(train_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=str,
                        default=",".join(f"{x:g}" for x in DEFAULT_LEVELS))
    parser.add_argument("--severities", type=str,
                        default=",".join(f"{x:g}" for x in DEFAULT_SEVERITIES))
    parser.add_argument("--seeds", type=str,
                        default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/grid")
    parser.add_argument("--array-id", type=int, default=None)
    args = parser.parse_args()

    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    severities = [float(x) for x in args.severities.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = build_tasks(levels, severities, seeds)
    print(f"Total tasks: {len(tasks)} ({len(levels)} levels × {len(severities)} severities × {len(seeds)} seeds)")

    if args.array_id is not None:
        if not (0 <= args.array_id < len(tasks)):
            raise SystemExit(f"--array-id {args.array_id} out of range [0, {len(tasks)})")
        level, severity, seed = tasks[args.array_id]
        print(f"[array {args.array_id}] level={level} severity={severity} seed={seed}")
        run_one(level, severity, seed, args.output_dir, args.max_steps)
        return

    for i, (level, severity, seed) in enumerate(tasks):
        print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] level={level} severity={severity} seed={seed}\n{'='*60}")
        run_one(level, severity, seed, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
