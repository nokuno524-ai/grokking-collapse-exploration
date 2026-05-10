import os
import json
import jsonlines
import pathlib
import pytest
import pandas as pd
import numpy as np

from src.analysis.parser import (
    parse_csv_log,
    parse_jsonl_log,
    scan_results_dir,
    load_experiment,
    detect_grokking_point,
    compute_collapse_metrics
)

def test_parse_csv_log(tmp_path):
    csv_file = tmp_path / "log.csv"
    csv_file.write_text("step,train_loss,val_acc\n1,0.5,0.1\n2,0.4,0.2\n")

    df = parse_csv_log(str(csv_file))
    assert len(df) == 2
    assert "step" in df.columns
    assert "train_loss" in df.columns
    assert df["val_acc"].iloc[1] == 0.2

def test_parse_jsonl_log(tmp_path):
    jsonl_file = tmp_path / "log.jsonl"
    with jsonlines.open(str(jsonl_file), mode='w') as writer:
        writer.write({"step": 1, "train_loss": 0.5, "val_acc": 0.1})
        writer.write({"step": 2, "train_loss": 0.4, "val_acc": 0.2})

    df = parse_jsonl_log(str(jsonl_file))
    assert len(df) == 2
    assert "step" in df.columns
    assert df["val_acc"].iloc[1] == 0.2

def test_detect_grokking_point():
    df = pd.DataFrame({
        "step": [100, 200, 300, 400],
        "val_acc": [0.1, 0.4, 0.92, 0.98]
    })

    # Threshold 0.9
    assert detect_grokking_point(df, threshold=0.9) == 300

    # Threshold 0.95
    assert detect_grokking_point(df, threshold=0.95) == 400

    # Threshold 1.0
    assert detect_grokking_point(df, threshold=1.0) == -1

    # Custom column
    df_custom = pd.DataFrame({"step": [1, 2], "test_acc": [0.5, 0.99]})
    assert detect_grokking_point(df_custom, acc_col="test_acc", threshold=0.9) == 2

def test_compute_collapse_metrics():
    df = pd.DataFrame({
        "step": [1, 2, 3],
        "weight_norm": [10.0, 8.0, 5.0],
        "embedding_rank": [100.0, 90.0, 50.0],
        "grad_norm": [1.0, 0.5, 0.1]
    })

    metrics = compute_collapse_metrics(df)
    assert "weight_norm_reduction" in metrics
    # (10 - 5) / 10 = 0.5
    assert metrics["weight_norm_reduction"] == 0.5
    assert metrics["final_representation_rank"] == 50.0
    assert metrics["final_grad_norm"] == 0.1
    # mean of [1.0, 0.5, 0.1] = 1.6 / 3
    assert np.isclose(metrics["mean_grad_norm"], 1.6 / 3)

def test_scan_and_load(tmp_path):
    res_dir = tmp_path / "results"
    res_dir.mkdir()

    cond_dir = res_dir / "pure"
    cond_dir.mkdir()

    results_file = cond_dir / "results.json"
    results_file.write_text(json.dumps({
        "config": {"collapse_level": 0.0},
        "grokked": True,
        "final_test_acc": 0.99,
        "history": [{"step": 1, "val_acc": 0.99}]
    }))

    # test scan
    catalog = scan_results_dir(str(res_dir))
    assert len(catalog) == 1
    assert catalog[0]["condition_name"] == "pure"
    assert catalog[0]["grokked"] == True
    assert catalog[0]["has_history"] == True

    # test load
    data = load_experiment("pure", str(res_dir))
    assert data["grokked"] == True
    assert len(data["history"]) == 1
