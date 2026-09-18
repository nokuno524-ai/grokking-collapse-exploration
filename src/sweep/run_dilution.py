"""
Dilution sweep: Sweep mixture ratios (fraction of synthetic/collapsed data) across multiple seeds.
"""

import argparse
from pathlib import Path
from src.train import TrainConfig, train

DEFAULT_SEEDS = [42, 43, 44]
COLLAPSE_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/dilution")
    parser.add_argument("--severity", type=float, default=0.5, help="Collapse severity to use for mixing")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        for collapse_level in COLLAPSE_LEVELS:
            cond_name = f"dilution_r{int(collapse_level*100)}"
            print(f"\n{'='*60}\nRunning seed {seed}, condition {cond_name}\n{'='*60}")

            config = TrainConfig(
                prime=59,
                train_fraction=0.3,
                collapse_level=collapse_level,
                collapse_severity=args.severity,
                seed=seed,
                condition_name=cond_name,
                output_dir=str(Path(args.output_dir) / str(seed)),
                max_steps=args.max_steps,
            )
            train(config)

if __name__ == "__main__":
    main()
