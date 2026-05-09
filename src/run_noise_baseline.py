"""
Random label-noise baseline.

Distinguishes "collapse" (temperature-warped frequency distribution) from
"random label noise" (uniform random wrong labels) at matched fractions.

Fractions: [0.0, 0.05, 0.15, 0.30, 0.50]
Seeds:    [42, 43, 44, 45, 46]
=> 5 × 5 = 25 runs.

Layout:
  results/noise_baseline/noise<F>/seed_<seed>/results.json
"""

import argparse
from pathlib import Path

try:
    from .train import TrainConfig, train
except ImportError:
    from train import TrainConfig, train


DEFAULT_FRACTIONS = [0.0, 0.05, 0.15, 0.30, 0.50]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def fmt_dir(noise_fraction: float) -> str:
    return f"noise{noise_fraction:g}"


def build_tasks(fractions, seeds):
    return [(f, s) for f in fractions for s in seeds]


def run_one(noise_fraction: float, seed: int, output_root: str, max_steps: int):
    train_config = TrainConfig(
        prime=59,
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=1.0,
        collapse_level=0.0,
        collapse_severity=0.5,
        noise_fraction=noise_fraction,
        seed=seed,
        condition_name=f"seed_{seed}",
        output_dir=str(Path(output_root) / fmt_dir(noise_fraction)),
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
    parser.add_argument("--output-dir", type=str, default="results/noise_baseline")
    parser.add_argument("--array-id", type=int, default=None)
    args = parser.parse_args()

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = build_tasks(fractions, seeds)
    print(f"Total tasks: {len(tasks)} ({len(fractions)} fractions × {len(seeds)} seeds)")

    if args.array_id is not None:
        if not (0 <= args.array_id < len(tasks)):
            raise SystemExit(f"--array-id {args.array_id} out of range [0, {len(tasks)})")
        noise_fraction, seed = tasks[args.array_id]
        print(f"[array {args.array_id}] noise_fraction={noise_fraction} seed={seed}")
        run_one(noise_fraction, seed, args.output_dir, args.max_steps)
        return

    for i, (noise_fraction, seed) in enumerate(tasks):
        print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] noise_fraction={noise_fraction} seed={seed}\n{'='*60}")
        run_one(noise_fraction, seed, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
