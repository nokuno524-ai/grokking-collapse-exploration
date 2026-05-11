import json
import pytest
import os
import urllib.parse
from tools.experiment_utils import (
    ExperimentConfig,
    generate_sweep,
    aggregate_results,
    compare_conditions,
    find_best_condition,
)


def test_experiment_config_serialization():
    config_dict = {
        "name": "test_exp",
        "collapse_level": 0.5,
        "model_size": 128,
        "dataset": "arithmetic",
        "learning_rate": 0.001,
        "seed": 42,
    }
    config = ExperimentConfig.from_dict(config_dict)
    assert config.name == "test_exp"
    assert config.collapse_level == 0.5

    out_dict = config.to_dict()
    assert out_dict == config_dict


def test_experiment_config_to_filename():
    config = ExperimentConfig(
        name="test",
        collapse_level=0.1,
        model_size=64,
        dataset="ds",
        learning_rate=0.01,
        seed=1
    )
    filename = config.to_filename()
    expected = urllib.parse.quote("name=test_collapse=0.1_size=64_data=ds_lr=0.01_seed=1", safe="=-_")
    assert filename == expected


def test_generate_sweep():
    base_dict = {
        "name": "sweep",
        "collapse_level": 0.0,
        "model_size": 128,
        "dataset": "arithmetic",
        "learning_rate": 0.001,
        "seed": 0,
    }
    configs = generate_sweep(base_dict, "seed", [1, 2, 3])
    assert len(configs) == 3
    assert configs[0].seed == 1
    assert configs[1].seed == 2
    assert configs[2].seed == 3
    assert configs[0].name == "sweep"


def test_aggregate_results(tmp_path):
    # Create two groups of results, with different seeds
    group1_dir = tmp_path / "group1"
    group1_dir.mkdir()

    group2_dir = tmp_path / "group2"
    group2_dir.mkdir()

    config1 = {"name": "c1", "collapse_level": 0.0}
    config2 = {"name": "c2", "collapse_level": 0.5}

    # Group 1 seed 1
    with open(group1_dir / "results_s1.json", "w") as f:
        json.dump({
            "config": {**config1, "seed": 1},
            "final_test_acc": 0.9,
            "grokking_step": 1000,
            "final_weight_norm": 50,
            "history": [{"weight_norm": 10}]
        }, f)

    # Group 1 seed 2
    with open(group1_dir / "results_s2.json", "w") as f:
        json.dump({
            "config": {**config1, "seed": 2},
            "final_test_acc": 1.0,
            "grokking_step": 1200,
            "final_weight_norm": 60,
            "history": [{"weight_norm": 20}]
        }, f)

    # Group 2 seed 1
    with open(group2_dir / "results_s1.json", "w") as f:
        json.dump({
            "config": {**config2, "seed": 1},
            "final_test_acc": 0.5,
            "grokking_step": 5000,
            "final_weight_norm": 30,
            "history": [{"weight_norm": 5}]
        }, f)

    results = aggregate_results(str(tmp_path))

    assert len(results) == 2

    # Check group 1 values (c1, collapse_level 0.0)
    # the key is a stringified dict of config without seed
    str_key_1 = str({"collapse_level": 0.0, "name": "c1"})
    assert str_key_1 in results
    m1 = results[str_key_1]["metrics"]

    # mean acc: (0.9 + 1.0) / 2 = 0.95
    assert m1["final_accuracy"]["mean"] == pytest.approx(0.95)
    # mean grokking: (1000 + 1200) / 2 = 1100
    assert m1["grokking_step"]["mean"] == pytest.approx(1100)
    # wn change s1: 50-10=40, s2: 60-20=40 -> mean 40
    assert m1["weight_norm_change"]["mean"] == pytest.approx(40.0)
    assert m1["num_seeds"] == 2


def test_compare_conditions():
    results = {
        "{'collapse_level': 0.0, 'name': 'pure'}": {
            "config_without_seed": {"name": "pure", "collapse_level": 0.0},
            "metrics": {
                "num_seeds": 3,
                "final_accuracy": {"mean": 0.99, "std": 0.01},
                "grokking_step": {"mean": 1000.0, "std": 50.0},
                "weight_norm_change": {"mean": 20.0, "std": 2.0}
            }
        }
    }

    table = compare_conditions(results)
    assert "| pure (collapse=0.0) | 3 | 0.9900 ± 0.0100 | 1000.0 ± 50.0 | 20.0000 ± 2.0000 |" in table


def test_find_best_condition():
    results = {
        "pure": {
            "metrics": {
                "final_accuracy": {"mean": 0.99, "std": 0.01},
                "grokking_step": {"mean": 1000.0, "std": 50.0},
            }
        },
        "collapse": {
            "metrics": {
                "final_accuracy": {"mean": 0.50, "std": 0.1},
                "grokking_step": {"mean": 5000.0, "std": 500.0},
            }
        }
    }

    best_acc_key, _ = find_best_condition(results, "final_accuracy")
    assert best_acc_key == "pure"

    best_grok_key, _ = find_best_condition(results, "grokking_step")
    assert best_grok_key == "pure"
