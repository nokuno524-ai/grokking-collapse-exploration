import argparse
import os
import json
import uuid
from pathlib import Path
import datetime
import subprocess

import torch
import numpy as np
import random

from src.management.config import ExperimentConfig, SweepConfig
from src.train import TrainConfig, train


def set_seed(seed: int):
    """Sets deterministic seeds for torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_experiment_id(prefix: str = "") -> str:
    """Generates a unique experiment ID using timestamp and a short UUID."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    if prefix:
        return f"{prefix}_{timestamp}_{short_uuid}"
    return f"{timestamp}_{short_uuid}"


def run_experiment(config: ExperimentConfig):
    """Runs a single experiment based on an ExperimentConfig."""
    # Ensure seed is set
    set_seed(config.training.seed)

    # Generate unique ID and directory
    exp_id = generate_experiment_id(config.experiment_name)
    out_dir = Path(config.output_dir) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the full config to the output directory
    with open(out_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"Running experiment: {exp_id}")
    print(f"Output directory: {out_dir}")

    # Map to TrainConfig
    train_config = TrainConfig(
        prime=config.model.prime,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        n_layers=config.model.n_layers,

        max_steps=config.training.max_steps,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        batch_size=config.training.batch_size,
        eval_every=config.training.eval_every,
        log_every=config.training.log_every,
        save_every=config.training.save_every,

        collapse_level=config.dataset.collapse_level,
        collapse_severity=config.dataset.collapse_severity,
        train_fraction=config.dataset.train_fraction,
        noise_fraction=config.dataset.noise_fraction,
        seed=config.training.seed,

        output_dir=str(out_dir),
        condition_name="run"  # Train script will nest this, but we already created a unique dir
    )

    # Run training
    state = train(train_config)
    print(f"Experiment {exp_id} completed.")
    return state


def main():
    parser = argparse.ArgumentParser(description="Run Grokking-Collapse experiments from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--sweep", action="store_true", help="Treat config as a SweepConfig")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if args.sweep:
        sweep_config = SweepConfig.from_yaml(args.config)
        configs = sweep_config.generate_configs()
        print(f"Generated {len(configs)} configurations from sweep.")
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Starting {config.experiment_name}")
            run_experiment(config)
    else:
        config = ExperimentConfig.from_yaml(args.config)
        run_experiment(config)


if __name__ == "__main__":
    main()
