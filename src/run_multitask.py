import os
import argparse
from pathlib import Path

from data import get_all_conditions
from train import TrainConfig, train

TASKS = [
    "modular_arithmetic",
    "group_multiplication",
    "binary_addition",
    "sparse_parity",
    "in_context_learning"
]
COLLAPSE_LEVELS = ["pure", "medium_collapse"]

def run_multitask(max_steps: int = 10000, output_dir: str = "results/multitask"):
    conditions = get_all_conditions()

    for task in TASKS:
        for condition_name in COLLAPSE_LEVELS:
            config_data = conditions[condition_name]
            out_dir = Path(output_dir) / task / condition_name
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running task={task}, condition={condition_name}")

            train_config = TrainConfig(
                prime=config_data.prime,
                train_fraction=config_data.train_fraction,
                collapse_level=config_data.collapse_level,
                collapse_severity=config_data.collapse_severity,
                noise_fraction=config_data.noise_fraction,
                seed=config_data.seed,
                condition_name=condition_name,
                output_dir=str(Path(output_dir) / task),
                max_steps=max_steps,
                task=task
            )
            train(train_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="results/multitask")
    args = parser.parse_args()
    run_multitask(args.max_steps, args.output_dir)
