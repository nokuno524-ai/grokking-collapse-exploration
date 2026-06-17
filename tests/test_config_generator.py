import json
from pathlib import Path
from tools.config_generator import ExperimentConfig, generate_sweep, generate_full_matrix, write_configs


def test_experiment_config_to_dict():
    config = ExperimentConfig(
        data_condition="pure",
        model_size=128,
        learning_rate=0.001,
        num_steps=50000,
        seed=42
    )
    d = config.to_dict()
    assert d == {
        "data_condition": "pure",
        "model_size": 128,
        "learning_rate": 0.001,
        "num_steps": 50000,
        "seed": 42
    }


def test_experiment_config_to_json():
    config = ExperimentConfig(
        data_condition="pure",
        model_size=128,
        learning_rate=0.001,
        num_steps=50000,
        seed=42
    )
    j = config.to_json()
    d = json.loads(j)
    assert d == {
        "data_condition": "pure",
        "model_size": 128,
        "learning_rate": 0.001,
        "num_steps": 50000,
        "seed": 42
    }


def test_experiment_config_from_dict():
    d = {
        "data_condition": "low_collapse",
        "model_size": 256,
        "learning_rate": 0.005,
        "num_steps": 10000,
        "seed": 100
    }
    config = ExperimentConfig.from_dict(d)
    assert config.data_condition == "low_collapse"
    assert config.model_size == 256
    assert config.learning_rate == 0.005
    assert config.num_steps == 10000
    assert config.seed == 100


def test_generate_sweep():
    base_config = ExperimentConfig(
        data_condition="pure",
        model_size=128,
        learning_rate=0.001,
        num_steps=50000,
        seed=42
    )
    seeds = [42, 43, 44]
    configs = generate_sweep(base_config, "seed", seeds)

    assert len(configs) == 3
    for i, seed in enumerate(seeds):
        assert configs[i].seed == seed
        assert configs[i].data_condition == "pure"
        assert configs[i].model_size == 128


def test_generate_full_matrix():
    params = {
        "data_condition": ["pure", "high_collapse"],
        "model_size": [128],
        "learning_rate": [0.001],
        "num_steps": [50000],
        "seed": [42, 43]
    }
    configs = generate_full_matrix(params)

    assert len(configs) == 4

    # Check that all combinations are present
    conditions = [c.data_condition for c in configs]
    seeds = [c.seed for c in configs]

    assert conditions.count("pure") == 2
    assert conditions.count("high_collapse") == 2
    assert seeds.count(42) == 2
    assert seeds.count(43) == 2


def test_write_configs(tmp_path):
    configs = [
        ExperimentConfig("pure", 128, 0.001, 50000, 42),
        ExperimentConfig("low_collapse", 128, 0.001, 50000, 42)
    ]

    output_dir = tmp_path / "configs"
    write_configs(configs, str(output_dir))

    assert output_dir.exists()
    assert (output_dir / "config_0.json").exists()
    assert (output_dir / "config_1.json").exists()

    with open(output_dir / "config_0.json", "r") as f:
        data = json.load(f)
        assert data["data_condition"] == "pure"

    with open(output_dir / "config_1.json", "r") as f:
        data = json.load(f)
        assert data["data_condition"] == "low_collapse"
