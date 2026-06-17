import json
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_condition: str
    model_size: int
    learning_rate: float
    num_steps: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**d)


def generate_sweep(base_config: ExperimentConfig, param: str, values: list) -> list[ExperimentConfig]:
    """Generates a list of configs by sweeping one parameter over a list of values."""
    configs = []
    base_dict = base_config.to_dict()
    for value in values:
        new_dict = base_dict.copy()
        new_dict[param] = value
        configs.append(ExperimentConfig.from_dict(new_dict))
    return configs


def generate_full_matrix(params: dict[str, list]) -> list[ExperimentConfig]:
    """Generates a cartesian product of parameters to create a full sweep."""
    keys = list(params.keys())
    values = list(params.values())

    configs = []
    for combination in itertools.product(*values):
        kwargs = dict(zip(keys, combination))
        configs.append(ExperimentConfig(**kwargs))

    return configs


def write_configs(configs: list[ExperimentConfig], output_dir: str) -> None:
    """Writes a list of configs to the specified directory, one JSON file per config."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, config in enumerate(configs):
        file_path = out_path / f"config_{i}.json"
        with open(file_path, "w") as f:
            f.write(config.to_json())


def main():
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    base_config = ExperimentConfig(
        data_condition="pure",
        model_size=128,
        learning_rate=0.001,
        num_steps=50000,
        seed=42
    )

    configs = generate_sweep(base_config, "data_condition", conditions)
    write_configs(configs, "generated_configs")
    print(f"Generated {len(configs)} configurations in 'generated_configs'")


if __name__ == "__main__":
    main()
