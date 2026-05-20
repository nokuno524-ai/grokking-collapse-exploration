from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    prime: int = 59  # Modular arithmetic modulus
    train_fraction: float = 0.3  # Fraction of data for training
    collapse_level: float = 0.0  # Fraction of training data replaced by synthetic
    collapse_severity: float = 0.5  # How much the synthetic generator has "collapsed" (0=fresh, 1=fully collapsed)
    noise_fraction: float = 0.0  # Fraction of training labels replaced with uniform random labels (baseline)
    seed: int = 42

    def __post_init__(self):
        assert self.prime > 0, "prime must be positive"
        assert 0.0 <= self.train_fraction <= 1.0, "train_fraction must be in [0, 1]"
        assert 0.0 <= self.collapse_level <= 1.0, "collapse_level must be in [0, 1]"
        assert 0.0 <= self.collapse_severity <= 1.0, "collapse_severity must be in [0, 1]"
        assert 0.0 <= self.noise_fraction <= 1.0, "noise_fraction must be in [0, 1]"

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
    train_fraction: float = 0.3
    noise_fraction: float = 0.0
    seed: int = 42

    # Output
    output_dir: str = "results"
    condition_name: str = "default"

    def __post_init__(self):
        assert self.prime > 0, "prime must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.lr > 0, "lr must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0.0 <= self.collapse_level <= 1.0, "collapse_level must be in [0, 1]"
        assert 0.0 <= self.collapse_severity <= 1.0, "collapse_severity must be in [0, 1]"
        assert 0.0 <= self.train_fraction <= 1.0, "train_fraction must be in [0, 1]"
        assert 0.0 <= self.noise_fraction <= 1.0, "noise_fraction must be in [0, 1]"

@dataclass
class TrainState:
    """Tracks training state and metrics."""
    step: int = 0
    train_loss: float = float('inf')
    test_loss: float = float('inf')
    train_acc: float = 0.0
    test_acc: float = 0.0
    weight_norm: float = 0.0
    embedding_rank: float = 0.0
    fourier_concentration: float = 0.0
    grokked: bool = False
    grokking_step: Optional[int] = None
    grokking_threshold: float = 0.95
    history: List[dict] = field(default_factory=list)
