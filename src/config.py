from dataclasses import dataclass

@dataclass
class TrainConfig:
    """Centralized training configuration for all experiments."""
    # Model architecture
    prime: int = 59
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 1

    # Optimization and Training loop
    max_steps: int = 50000
    lr: float = 1e-3
    weight_decay: float = 1.0  # Crucial for grokking!
    batch_size: int = 512

    # Logging and checkpoints
    eval_every: int = 100
    log_every: int = 50
    save_every: int = 5000

    # Dataset and collapse settings
    collapse_level: float = 0.0
    collapse_severity: float = 0.5
    train_fraction: float = 0.3
    noise_fraction: float = 0.0
    seed: int = 42

    # Input/Output paths
    output_dir: str = "results"
    condition_name: str = "default"
