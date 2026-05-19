"""
Experiment C — weight-decay x noise-rate grid (toy modular arithmetic).

Tests the theoretical conjecture that the noise-rate threshold eta* shifts
left (higher noise tolerance) as weight-decay lambda grows.

Grid (default): wd in {0.3, 1.0, 3.0} x noise in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30} x 5 seeds
            => 3 * 6 * 5 = 90 runs (toy, ~3-5 minutes each)

Layout:
  results/exp_c_grid/wd<W>/noise<N>/seed_<S>/results.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

try:
    from .train import TrainConfig, train
except ImportError:
    from train import TrainConfig, train  # type: ignore


DEFAULT_WDS = [0.3, 1.0, 3.0]
DEFAULT_NOISES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def fmt_wd(wd: float) -> str:
    return f"wd{wd:g}"


def fmt_noise(n: float) -> str:
    return f"noise{n:g}"


def build_tasks(wds, noises, seeds) -> List[Tuple[float, float, int]]:
    return [(w, n, s) for w in wds for n in noises for s in seeds]


def run_one(wd: float, noise_fraction: float, seed: int,
            output_root: str, max_steps: int):
    out_dir = Path(output_root) / fmt_wd(wd) / fmt_noise(noise_fraction)
    train_config = TrainConfig(
        prime=59,
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=wd,
        collapse_level=0.0,
        collapse_severity=0.5,
        noise_fraction=noise_fraction,
        seed=seed,
        condition_name=f"seed_{seed}",
        output_dir=str(out_dir),
        max_steps=max_steps,
    )
    return train(train_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-decays", type=str,
                        default=",".join(f"{x:g}" for x in DEFAULT_WDS))
    parser.add_argument("--noise-fractions", type=str,
                        default=",".join(f"{x:g}" for x in DEFAULT_NOISES))
    parser.add_argument("--seeds", type=str,
                        default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str,
                        default="results/exp_c_grid")
    parser.add_argument("--array-id", type=int, default=None,
                        help="If set, run only the task at this index.")
    parser.add_argument("--print-task-count", action="store_true",
                        help="Just print the total number of tasks and exit.")
    args = parser.parse_args()

    wds = [float(x) for x in args.weight_decays.split(",") if x]
    noises = [float(x) for x in args.noise_fractions.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    tasks = build_tasks(wds, noises, seeds)

    if args.print_task_count:
        print(len(tasks))
        return

    if args.array_id is not None:
        if not 0 <= args.array_id < len(tasks):
            raise SystemExit(
                f"--array-id {args.array_id} out of range [0, {len(tasks)})"
            )
        wd, noise, seed = tasks[args.array_id]
        print(f"[array {args.array_id}/{len(tasks)}] wd={wd} noise={noise} seed={seed}")
        run_one(wd, noise, seed, args.output_dir, args.max_steps)
        return

    for i, (wd, noise, seed) in enumerate(tasks):
        print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] wd={wd} noise={noise} seed={seed}\n{'='*60}")
        run_one(wd, noise, seed, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
