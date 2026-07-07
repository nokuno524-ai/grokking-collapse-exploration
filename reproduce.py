import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.train import train, TrainConfig, get_all_conditions

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_reproduction_sweep(seeds, max_steps, output_dir):
    """Run all conditions across multiple seeds."""
    conditions = get_all_conditions()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"RUNNING SEED: {seed}")
        print(f"{'='*80}")

        # Ensure seed is set at the start of each run
        set_seed(seed)

        for name, data_config in conditions.items():
            print(f"\n--- Condition: {name} | Seed: {seed} ---")

            # The config object uses condition_name as part of the output path
            # We want to structure it as output_dir/condition_name/seed_X
            seed_condition_name = f"{name}/seed_{seed}"

            # Override data config seed just to be explicit
            data_config.seed = seed

            train_config = TrainConfig(
                collapse_level=data_config.collapse_level,
                collapse_severity=data_config.collapse_severity,
                condition_name=seed_condition_name,
                output_dir=str(out_path),
                max_steps=max_steps,
                seed=seed,
                # Using somewhat shorter evaluation frequency for reproduction runs to be safe
                # though these default to 100 in TrainConfig.
            )

            train(train_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Grokking/Collapse experiments.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="List of seeds to run")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Max training steps per run. NOTE: set to 50000 for full reproduction.")
    parser.add_argument("--output-dir", type=str, default="results/reproduce",
                        help="Directory to save results")
    args = parser.parse_args()

    run_reproduction_sweep(args.seeds, args.max_steps, args.output_dir)
