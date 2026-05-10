import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import yaml

@dataclass
class CollapseConfig:
    """Configuration for collapse injection."""
    collapse_type: str = "synthetic_data_ratio" # e.g., synthetic_data_ratio, weight_noise, gradient_noise, data_repetition
    severity: str = "none" # none, low, medium, severe
    injection_point: str = "data" # data, model, optimizer

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    model_type: str = "transformer"
    dataset: str = "modular_arithmetic"
    collapse_level: float = 0.0
    collapse_type: str = "synthetic_data_ratio"
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    epochs: int = 50000
    batch_size: int = 512
    seed: int = 42
    log_interval: int = 100
    # Additional components for generic configuration support
    collapse_config: Optional[CollapseConfig] = None
    output_dir: str = "results"

    def __post_init__(self):
        if self.collapse_config is None:
            # Determine severity based on collapse_level
            severity = "none"
            if self.collapse_level > 0.4:
                severity = "severe"
            elif self.collapse_level > 0.15:
                severity = "high"
            elif self.collapse_level > 0.05:
                severity = "medium"
            elif self.collapse_level > 0:
                severity = "low"
            self.collapse_config = CollapseConfig(
                collapse_type=self.collapse_type,
                severity=severity,
                injection_point="data"
            )
        elif isinstance(self.collapse_config, dict):
            self.collapse_config = CollapseConfig(**self.collapse_config)

def load_config(path: str) -> ExperimentConfig:
    """Load config from a YAML or JSON file."""
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return ExperimentConfig(**data)

def save_config(config: ExperimentConfig, path: str):
    """Save config to a YAML or JSON file."""
    data = asdict(config)
    with open(path, "w") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            yaml.dump(data, f)
        else:
            json.dump(data, f, indent=2)

def get_default_configs() -> List[ExperimentConfig]:
    """Generate a standard experiment grid."""
    configs = []
    conditions = [
        ("pure", 0.0, "none"),
        ("low_collapse", 0.05, "low"),
        ("medium_collapse", 0.15, "medium"),
        ("high_collapse", 0.30, "severe"), # Based on dataset defaults mapping roughly to standard severities
        ("severe_collapse", 0.50, "severe"),
    ]
    for name, level, severity in conditions:
        collapse_config = CollapseConfig(collapse_type="synthetic_data_ratio", severity=severity, injection_point="data")
        configs.append(ExperimentConfig(
            collapse_level=level,
            collapse_config=collapse_config,
            output_dir=f"results/{name}"
        ))
    return configs

def validate_config(config: ExperimentConfig) -> List[str]:
    """Validate experiment configuration parameters."""
    errors = []
    if config.collapse_level < 0 or config.collapse_level > 1:
        errors.append("collapse_level must be between 0 and 1.")
    if config.learning_rate <= 0:
        errors.append("learning_rate must be positive.")
    if config.weight_decay < 0:
        errors.append("weight_decay must be non-negative.")
    if config.epochs <= 0:
        errors.append("epochs must be positive.")
    if config.batch_size <= 0:
        errors.append("batch_size must be positive.")

    if config.collapse_config:
        valid_types = ["synthetic_data_ratio", "weight_noise", "gradient_noise", "data_repetition"]
        if config.collapse_config.collapse_type not in valid_types:
            errors.append(f"Invalid collapse_type: {config.collapse_config.collapse_type}. Must be one of {valid_types}.")
        valid_severities = ["none", "low", "medium", "severe", "high"] # Added high to handle legacy mappings smoothly
        if config.collapse_config.severity not in valid_severities:
            errors.append(f"Invalid severity: {config.collapse_config.severity}. Must be one of {valid_severities}.")
        valid_injection = ["data", "model", "optimizer"]
        if config.collapse_config.injection_point not in valid_injection:
            errors.append(f"Invalid injection_point: {config.collapse_config.injection_point}. Must be one of {valid_injection}.")

    return errors
