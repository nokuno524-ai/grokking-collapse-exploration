"""
Curriculum schedule sweep: Test ordered exposure effects on grokking.
Runs clean->collapse, collapse->clean, and interleaved schedules.
"""

import argparse
from pathlib import Path
from src.train import TrainConfig, train

DEFAULT_SEEDS = [42, 43, 44]
SCHEDULES = ["none", "clean_to_collapse", "collapse_to_clean", "interleaved"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/curriculum")
    parser.add_argument("--switch-step", type=int, default=25000)
    parser.add_argument("--collapse-level", type=float, default=0.15)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        for mode in SCHEDULES:
            print(f"\n{'='*60}\nRunning seed {seed}, curriculum {mode}\n{'='*60}")

            config = TrainConfig(
                prime=59,
                train_fraction=0.3,
                collapse_level=args.collapse_level,
                curriculum_mode=mode,
                switch_step=args.switch_step,
                interleaved_ratio=0.5,
                seed=seed,
                condition_name=mode,
                output_dir=str(Path(args.output_dir) / str(seed)),
                max_steps=args.max_steps,
            )
            train(config)

if __name__ == "__main__":
    main()
