import json
from src.management.config import ExperimentConfig, SweepConfig
from src.management.results import ResultsCollector
from run_experiment import generate_experiment_id


def test_config_loading(tmp_path):
    yaml_content = """
    experiment_name: "test_exp"
    model:
      prime: 97
    dataset:
      collapse_level: 0.2
    training:
      max_steps: 1000
    """
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    config = ExperimentConfig.from_yaml(str(config_file))
    assert config.experiment_name == "test_exp"
    assert config.model.prime == 97
    assert config.dataset.collapse_level == 0.2
    assert config.training.max_steps == 1000
    assert config.model.d_model == 128  # default


def test_sweep_config(tmp_path):
    yaml_content = """
    experiment_name: "sweep_test"
    base_config:
      model:
        prime: 59
    sweep_params:
      training.weight_decay: [0.1, 1.0]
      dataset.collapse_level: [0.0, 0.5]
    """
    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(yaml_content)

    sweep = SweepConfig.from_yaml(str(config_file))
    configs = sweep.generate_configs()

    assert len(configs) == 4

    wds = [c.training.weight_decay for c in configs]
    assert sorted(wds) == [0.1, 0.1, 1.0, 1.0]

    clvls = [c.dataset.collapse_level for c in configs]
    assert sorted(clvls) == [0.0, 0.0, 0.5, 0.5]


def test_experiment_id():
    id1 = generate_experiment_id("test")
    id2 = generate_experiment_id("test")

    assert id1.startswith("test_")
    assert id1 != id2


def test_results_aggregation(tmp_path):
    # Setup mock results structure
    run1_dir = tmp_path / "run1"
    run1_dir.mkdir()
    run1_data = {
        "config": {
            "experiment_name": "run1",
            "dataset": {"collapse_level": 0.0},
            "training": {"weight_decay": 1.0}
        },
        "final_test_acc": 0.95,
        "grokked": True,
        "grokking_step": 1000
    }
    with open(run1_dir / "results.json", "w") as f:
        json.dump(run1_data, f)

    run2_dir = tmp_path / "run2"
    run2_dir.mkdir()
    run2_data = {
        "config": {
            "experiment_name": "run2",
            "dataset": {"collapse_level": 0.5},
            "training": {"weight_decay": 1.0}
        },
        "final_test_acc": 0.10,
        "grokked": False,
        "grokking_step": None
    }
    with open(run2_dir / "results.json", "w") as f:
        json.dump(run2_data, f)

    collector = ResultsCollector(str(tmp_path))
    df = collector.aggregate_to_dataframe()

    assert len(df) == 2
    assert "experiment_name" in df.columns
    assert "dataset.collapse_level" in df.columns
    assert "final_test_acc" in df.columns

    # Sort to ensure predictable order
    df = df.sort_values("experiment_name").reset_index(drop=True)

    assert df.loc[0, "dataset.collapse_level"] == 0.0
    assert df.loc[0, "final_test_acc"] == 0.95
    assert df.loc[0, "grokked"] is True

    assert df.loc[1, "dataset.collapse_level"] == 0.5
    assert df.loc[1, "final_test_acc"] == 0.10
    assert df.loc[1, "grokked"] is False
