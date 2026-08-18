import pytest
import json
import pandas as pd
from pathlib import Path
from src.analysis.results_analysis import parse_results_dir

def test_parse_results_dir(tmp_path):
    # Setup mock results
    run1 = tmp_path / "run1"
    run1.mkdir()

    mock_data_1 = {
        "config": {"condition_name": "pure", "collapse_level": 0.0, "seed": 42},
        "history": [
            {"step": 100, "test_acc": 0.1, "train_acc": 0.5, "weight_norm": 10.0},
            {"step": 200, "test_acc": 0.99, "train_acc": 0.99, "weight_norm": 12.0},
            {"step": 300, "test_acc": 0.99, "train_acc": 1.0, "weight_norm": 15.0},
        ],
        "final_test_acc": 0.99
    }

    with open(run1 / "results.json", "w") as f:
        json.dump(mock_data_1, f)

    run2 = tmp_path / "run2"
    run2.mkdir()

    # Run that doesn't grok
    mock_data_2 = {
        "config": {"condition_name": "severe", "collapse_level": 0.5, "seed": 42},
        "history": [
            {"step": 100, "test_acc": 0.1, "train_acc": 0.5, "weight_norm": 10.0},
            {"step": 200, "test_acc": 0.1, "train_acc": 0.6, "weight_norm": 12.0},
            {"step": 300, "test_acc": 0.1, "train_acc": 0.7, "weight_norm": 15.0},
        ]
    }

    with open(run2 / "results.json", "w") as f:
        json.dump(mock_data_2, f)

    df = parse_results_dir(tmp_path, window_size=2)

    assert len(df) == 2

    pure_row = df[df.condition == "pure"].iloc[0]
    assert pure_row.grokked_detected == True
    assert pure_row.grokking_step_detected == 200

    severe_row = df[df.condition == "severe"].iloc[0]
    assert severe_row.grokked_detected == False
    assert severe_row.grokking_step_detected == -1

    # Test Empty history
    run3 = tmp_path / "run3"
    run3.mkdir()
    with open(run3 / "results.json", "w") as f:
        json.dump({"config": {"condition_name": "empty"}, "history": []}, f)

    df2 = parse_results_dir(tmp_path, window_size=2)
    assert len(df2) == 2  # The empty one is skipped
