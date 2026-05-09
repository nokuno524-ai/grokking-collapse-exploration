"""
Data-scarcity baseline.

Reduce the training fraction to the *effective clean count* of each collapse condition,
without corrupting any labels. This isolates "less data" from "corrupted data."

Mapping (matches collapse_level in {0, 0.05, 0.15, 0.30, 0.50}, base train_fraction=0.30):
  0.30 = 0.30 * (1 - 0.00)
  0.285 = 0.30 * (1 - 0.05)
  0.255 = 0.30 * (1 - 0.15)
  0.21  = 0.30 * (1 - 0.30)
  0.15  = 0.30 * (1 - 0.50)

Seeds: [42, 43, 44, 45, 46] => 5 × 5 = 25 runs.

Layout:
  results/scarcity_baseline/frac<F>/seed_<seed>/results.json
"""

import argparse
from pathlib import Path

try:
    from .train import TrainConfig, train
except ImportError:
    from train import TrainConfig, train


DEFAULT_FRACTIONS = [0.30, 0.285, 0.255, 0.21, 0.15]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def fmt_dir(train_fraction: float) -> str:
    return f"frac{train_fraction:g}"


def build_tasks(fractions, seeds):
    return [(f, s) for f in fractions for s in seeds]


def run_one(train_fraction: float, seed: int, output_root: str, max_steps: int):
    train_config = TrainConfig(
        prime=59,
        train_fraction=train_fraction,
        lr=1e-3,
        weight_decay=1.0,
        collapse_level=0.0,
        collapse_severity=0.5,
        noise_fraction=0.0,
        seed=seed,
        condition_name=f"seed_{seed}",
        output_dir=str(Path(output_root) / fmt_dir(train_fraction)),
        max_steps=max_steps,
    )
    return train(train_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=str,
                        default=",".join(f"{x:g}" for x in DEFAULT_FRACTIONS))
    parser.add_argument("--seeds", type=str,
                        default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/scarcity_baseline")
    parser.add_argument("--array-id", type=int, default=None)
    args = parser.parse_args()

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = build_tasks(fractions, seeds)
    print(f"Total tasks: {len(tasks)} ({len(fractions)} fractions × {len(seeds)} seeds)")

    if args.array_id is not None:
        if not (0 <= args.array_id < len(tasks)):
            raise SystemExit(f"--array-id {args.array_id} out of range [0, {len(tasks)})")
        train_fraction, seed = tasks[args.array_id]
        print(f"[array {args.array_id}] train_fraction={train_fraction} seed={seed}")
        run_one(train_fraction, seed, args.output_dir, args.max_steps)
        return

    for i, (train_fraction, seed) in enumerate(tasks):
        print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] train_fraction={train_fraction} seed={seed}\n{'='*60}")
        run_one(train_fraction, seed, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
