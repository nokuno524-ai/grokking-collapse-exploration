from dataclasses import dataclass, field
from typing import Optional
import yaml
import os

@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    prime: int = 59
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 1

    # Training
    max_steps: int = 50000
    lr: float = 1e-3
    weight_decay: float = 1.0  # Key hyperparameter for grokking!
    batch_size: int = 512

    # Logging
    eval_every: int = 100
    log_every: int = 50
    save_every: int = 5000

    # Data
    collapse_level: float = 0.0
    collapse_severity: float = 0.5
    seed: int = 42

    # Output
    output_dir: str = "results"
    condition_name: str = "default"

def load_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)
