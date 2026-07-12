import pytest
import yaml
import json
import os
import pandas as pd
from pathlib import Path
from runner import load_config
from aggregate_results import aggregate_metrics

def test_config_parsing(tmp_path):
    """Test that runner can parse the yaml configuration format."""
    config_data = {
        "model": {"layers": 2},
        "data": {"collapse_ratio": 0.15},
        "training": {"steps": 100}
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    loaded_config = load_config(str(config_file))

    assert loaded_config["model"]["layers"] == 2
    assert loaded_config["data"]["collapse_ratio"] == 0.15
    assert loaded_config["training"]["steps"] == 100

def test_result_aggregation(tmp_path):
    """Test that result aggregation groups correctly and handles missing values."""
    dir1 = tmp_path / "run1"
    dir1.mkdir()

    dir2 = tmp_path / "run2"
    dir2.mkdir()

    dir3 = tmp_path / "run3"
    dir3.mkdir()

    # Run 1: Normal grokking run
    res1 = {
        "config": {
            "data": {"collapse_ratio": 0.0, "noise_fraction": 0.0},
            "training": {"weight_decay": 1.0, "seed": 42}
        },
        "final_test_acc": 0.98,
        "final_weight_norm": 20.0,
        "final_fourier_concentration": 0.15,
        "grokked": True,
        "grokking_step": 2000
    }

    # Run 2: Same condition, different seed, also grokked
    res2 = {
        "config": {
            "data": {"collapse_ratio": 0.0, "noise_fraction": 0.0},
            "training": {"weight_decay": 1.0, "seed": 43}
        },
        "final_test_acc": 0.99,
        "final_weight_norm": 21.0,
        "final_fourier_concentration": 0.16,
        "grokked": True,
        "grokking_step": 3000
    }

    # Run 3: Different condition (collapse), did not grok
    res3 = {
        "config": {
            "data": {"collapse_ratio": 0.5, "noise_fraction": 0.0},
            "training": {"weight_decay": 1.0, "seed": 42}
        },
        "final_test_acc": 0.02,
        "final_weight_norm": 45.0,
        "final_fourier_concentration": 0.01,
        "grokked": False,
        "grokking_step": None
    }

    files = []
    for i, res in enumerate([res1, res2, res3], 1):
        p = tmp_path / f"run{i}" / "results.json"
        with open(p, "w") as f:
            json.dump(res, f)
        files.append(str(p))

    df = aggregate_metrics(files)

    assert len(df) == 3

    # Test grouping behavior that aggregate_results uses
    group_cols = ["collapse_ratio", "noise_fraction", "weight_decay"]

    # Mean of run1 and run2
    pure_mean = df[df["collapse_ratio"] == 0.0]["grokking_step"].mean()
    assert pure_mean == 2500.0

    # Make sure run3 handled None properly
    collapsed = df[df["collapse_ratio"] == 0.5].iloc[0]
    assert collapsed["grokked"] == False
    assert collapsed["grokking_step"] == float('inf')
