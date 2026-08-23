import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    """Unified configuration for grokking experiments."""
    # Task settings
    task: str = "modular_arithmetic"
    prime: int = 59
    train_fraction: float = 0.3

    # Model settings
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 1
    dropout: float = 0.0

    # Optimizer settings
    lr: float = 1e-3
    weight_decay: float = 1.0
    batch_size: int = 512
    max_steps: int = 50000

    # Collapse / baseline settings
    collapse_level: float = 0.0
    collapse_severity: float = 0.5
    noise_fraction: float = 0.0
    seed: int = 42

    # Logging
    eval_every: int = 100
    log_every: int = 50
    save_every: int = 5000
    output_dir: str = "results"
    condition_name: str = "default"

def load_config(path: str) -> ExperimentConfig:
    """Load config from YAML file and validate."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if not data:
        return ExperimentConfig()

    valid_tasks = ["modular_arithmetic", "polynomial_identity", "sparse_parity", "digit_sorting"]
    if "task" in data and data["task"] not in valid_tasks:
        raise ValueError(f"Invalid task: {data['task']}. Must be one of {valid_tasks}")

    # Only pass valid kwargs
    valid_keys = ExperimentConfig.__dataclass_fields__.keys()
    kwargs = {k: v for k, v in data.items() if k in valid_keys}

    return ExperimentConfig(**kwargs)
