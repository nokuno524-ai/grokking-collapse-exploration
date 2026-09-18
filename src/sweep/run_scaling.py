"""
Dataset size scaling sweep crossed with collapse severity.
"""

import argparse
from pathlib import Path
from src.train import TrainConfig, train

DEFAULT_SEEDS = [42, 43, 44]
TRAIN_FRACTIONS = [0.15, 0.3, 0.6, 1.2]
COLLAPSE_CONDITIONS = [
    ("pure", 0.0, 0.0),
    ("medium", 0.15, 0.5),
    ("severe", 0.50, 0.9),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/scaling")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        for frac in TRAIN_FRACTIONS:
            for cond_name, c_level, c_sev in COLLAPSE_CONDITIONS:
                run_name = f"frac_{frac}_{cond_name}"
                print(f"\n{'='*60}\nRunning seed {seed}, {run_name}\n{'='*60}")

                config = TrainConfig(
                    prime=59,
                    train_fraction=frac,
                    collapse_level=c_level,
                    collapse_severity=c_sev,
                    seed=seed,
                    condition_name=run_name,
                    output_dir=str(Path(args.output_dir) / str(seed)),
                    max_steps=args.max_steps,
                )
                train(config)

if __name__ == "__main__":
    main()
