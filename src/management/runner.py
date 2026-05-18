import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple

try:
    from .config import ExperimentConfig
    from src.train import TrainConfig, train
except ImportError:
    # Handle direct script execution vs module import
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.management.config import ExperimentConfig
    from src.train import TrainConfig, train


def build_tasks(config: ExperimentConfig) -> List[Tuple]:
    """Generates all combinations of parameters."""
    tasks = []
    for wd in config.weight_decays:
        for noise in config.noise_fractions:
            for c_level in config.collapse_levels:
                for c_sev in config.collapse_severities:
                    for seed in config.seeds:
                        tasks.append((wd, noise, c_level, c_sev, seed))
    return tasks


def run_task(config: ExperimentConfig, array_id: int):
    """Runs a specific task from the grid."""
    tasks = build_tasks(config)

    if array_id < 0 or array_id >= len(tasks):
        raise ValueError(f"Array ID {array_id} out of range [0, {len(tasks)-1}]")

    wd, noise, c_level, c_sev, seed = tasks[array_id]

    # Construct distinct output path for this specific run
    run_name = f"wd{wd:g}_n{noise:g}_c{c_level:g}_s{c_sev:g}_seed{seed}"
    out_dir = Path(config.output_dir) / run_name

    print(f"Running task {array_id} -> {run_name}")
    print(f"Output directory: {out_dir}")

    # Build TrainConfig
    train_config = TrainConfig(
        # Model
        prime=config.dataset.prime,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        n_layers=config.model.n_layers,

        # Training
        max_steps=config.training.max_steps,
        lr=config.training.lr,
        weight_decay=wd,
        batch_size=config.training.batch_size,

        # Data
        train_fraction=config.dataset.train_fraction,
        collapse_level=c_level,
        collapse_severity=c_sev,
        noise_fraction=noise,
        seed=seed,

        # Admin
        condition_name=run_name,
        output_dir=str(config.output_dir),  # train.py appends condition_name automatically

        # Logging
        eval_every=config.training.eval_every,
        log_every=config.training.log_every,
        save_every=config.training.save_every,
    )

    # Run training
    train(train_config)


def main():
    parser = argparse.ArgumentParser(description="Run an experiment task from a grid.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to ExperimentConfig YAML")
    parser.add_argument("--array-id", type=int, required=True, help="SLURM_ARRAY_TASK_ID or local index")

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    config = ExperimentConfig.load_yaml(args.config_path)
    run_task(config, args.array_id)


if __name__ == "__main__":
    main()
