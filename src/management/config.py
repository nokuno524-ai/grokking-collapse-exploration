import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import yaml
import json
import os

@dataclass
class ComputeConfig:
    """Compute requirements for a job."""
    partition: str = "gpu-a40,gpu-a6000"
    gpus: int = 1
    cpus_per_task: int = 4
    mem: str = "16G"
    time: str = "01:30:00"
    account: str = "zhangmlgroup"

@dataclass
class ModelConfig:
    """Model hyperparameters."""
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 1

@dataclass
class DatasetConfig:
    """Dataset generation parameters."""
    prime: int = 59
    train_fraction: float = 0.3
    collapse_level: float = 0.0
    collapse_severity: float = 0.5
    noise_fraction: float = 0.0

@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""
    max_steps: int = 50000
    lr: float = 1e-3
    weight_decay: float = 1.0
    batch_size: int = 512
    eval_every: int = 100
    log_every: int = 50
    save_every: int = 5000

@dataclass
class ExperimentConfig:
    """
    Unified experiment configuration system.
    Supports a grid of contamination parameters and seeds.
    """
    name: str
    output_dir: str

    # Grid parameters
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    weight_decays: List[float] = field(default_factory=lambda: [1.0])
    noise_fractions: List[float] = field(default_factory=lambda: [0.0])
    collapse_levels: List[float] = field(default_factory=lambda: [0.0])
    collapse_severities: List[float] = field(default_factory=lambda: [0.5])

    # Nested configurations
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentConfig':
        """Create from a dictionary, correctly handling nested dataclasses."""

        # Extract nested configs if present, else use empty dicts to trigger defaults
        compute_data = data.pop('compute', {})
        model_data = data.pop('model', {})
        dataset_data = data.pop('dataset', {})
        training_data = data.pop('training', {})

        return cls(
            compute=ComputeConfig(**compute_data),
            model=ModelConfig(**model_data),
            dataset=DatasetConfig(**dataset_data),
            training=TrainingConfig(**training_data),
            **data
        )

    def save_yaml(self, path: str):
        """Save configuration to a YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def get_num_tasks(self) -> int:
        """Returns the total number of tasks in the grid."""
        return (len(self.seeds) * len(self.weight_decays) *
                len(self.noise_fractions) * len(self.collapse_levels) *
                len(self.collapse_severities))
