import os
import argparse
from pathlib import Path

from data import get_all_conditions
from train import TrainConfig, train

WDS = [0.0, 0.001, 0.01, 0.1, 1.0]
COLLAPSE_LEVELS = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

def run_grid(max_steps: int = 10000, output_dir: str = "results/wd_phase_diagram"):
    conditions = get_all_conditions()

    for wd in WDS:
        for condition_name in COLLAPSE_LEVELS:
            config_data = conditions[condition_name]
            out_dir = Path(output_dir) / f"wd_{wd}" / condition_name
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running wd={wd}, condition={condition_name}")

            train_config = TrainConfig(
                prime=config_data.prime,
                train_fraction=config_data.train_fraction,
                collapse_level=config_data.collapse_level,
                collapse_severity=config_data.collapse_severity,
                noise_fraction=config_data.noise_fraction,
                seed=config_data.seed,
                condition_name=condition_name,
                output_dir=str(Path(output_dir) / f"wd_{wd}"),
                max_steps=max_steps,
                weight_decay=wd
            )
            train(train_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="results/wd_phase_diagram")
    args = parser.parse_args()
    run_grid(args.max_steps, args.output_dir)
