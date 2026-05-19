"""
Experiment configuration definitions.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml


@dataclass
class ModelConfig:
    prime: int = 59
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 1


@dataclass
class DatasetConfig:
    prime: int = 59
    train_fraction: float = 0.3
    collapse_level: float = 0.0
    collapse_severity: float = 0.5
    noise_fraction: float = 0.0


@dataclass
class TrainingConfig:
    max_steps: int = 50000
    lr: float = 1e-3
    weight_decay: float = 1.0
    batch_size: int = 512
    eval_every: int = 100
    log_every: int = 50
    save_every: int = 5000
    seed: int = 42


@dataclass
class ExperimentConfig:
    experiment_name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        model_cfg = ModelConfig(**data.get('model', {}))
        dataset_cfg = DatasetConfig(**data.get('dataset', {}))
        training_cfg = TrainingConfig(**data.get('training', {}))

        return cls(
            experiment_name=data.get('experiment_name', 'default'),
            model=model_cfg,
            dataset=dataset_cfg,
            training=training_cfg,
            output_dir=data.get('output_dir', 'results')
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'model': self.model.__dict__,
            'dataset': self.dataset.__dict__,
            'training': self.training.__dict__,
            'output_dir': self.output_dir
        }


@dataclass
class SweepConfig:
    experiment_name: str
    base_config: ExperimentConfig
    # e.g. {"training.weight_decay": [0.1, 1.0], "dataset.collapse_level": [0.0, 0.5]}
    sweep_params: Dict[str, List[Any]]

    @classmethod
    def from_yaml(cls, path: str) -> 'SweepConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        base = ExperimentConfig.from_dict(data.get('base_config', {}))
        return cls(
            experiment_name=data.get('experiment_name', 'sweep'),
            base_config=base,
            sweep_params=data.get('sweep_params', {})
        )

    def generate_configs(self) -> List[ExperimentConfig]:
        import itertools

        keys = list(self.sweep_params.keys())
        values = list(self.sweep_params.values())
        combinations = list(itertools.product(*values))

        configs = []
        for idx, combo in enumerate(combinations):
            import copy
            config = copy.deepcopy(self.base_config)

            combo_name_parts = []
            for i, key in enumerate(keys):
                val = combo[i]
                parts = key.split('.')

                # set the value
                if parts[0] == 'model':
                    setattr(config.model, parts[1], val)
                elif parts[0] == 'dataset':
                    setattr(config.dataset, parts[1], val)
                elif parts[0] == 'training':
                    setattr(config.training, parts[1], val)
                else:
                    setattr(config, parts[0], val)

                combo_name_parts.append(f"{parts[-1]}{val}")

            # append sweep config values to name
            config.experiment_name = f"{self.experiment_name}_{'_'.join(combo_name_parts)}_s{config.training.seed}"
            configs.append(config)

        return configs
