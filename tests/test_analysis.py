import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import tempfile
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parse_results import parse_results_dir, export_to_csv
from src.statistical_analysis import compute_confidence_intervals, bootstrap_ci, mann_whitney_u_test, compute_correlations

@pytest.fixture
def dummy_results_dir(tmp_path):
    """Create a temporary directory with dummy results.json files."""
    results_dir = tmp_path / "results"

    # Pure condition
    pure_dir = results_dir / "pure"
    pure_dir.mkdir(parents=True)
    with open(pure_dir / "results.json", "w") as f:
        json.dump({
            "config": {"condition_name": "pure", "seed": 42},
            "grokking_step": 1400,
            "final_test_acc": 1.0,
            "grokked": True,
            "history": [{"step": 100, "test_acc": 0.1}, {"step": 1500, "test_acc": 1.0}]
        }, f)

    # Collapse condition
    collapse_dir = results_dir / "collapse"
    collapse_dir.mkdir(parents=True)
    with open(collapse_dir / "results.json", "w") as f:
        json.dump({
            "config": {"condition_name": "collapse", "seed": 42, "collapse_level": 0.5},
            "grokking_step": None,
            "final_test_acc": 0.1,
            "grokked": False,
            "history": [{"step": 100, "test_acc": 0.1}, {"step": 1500, "test_acc": 0.1}]
        }, f)

    return results_dir

def test_parse_results(dummy_results_dir):
    """Test that parser correctly flattens json and extracts metrics."""
    parsed = parse_results_dir(dummy_results_dir)
    assert len(parsed) == 2

    # Check that config is flattened
    pure_run = next(r for r in parsed if r["config_condition_name"] == "pure")
    assert pure_run["grokking_step"] == 1400
    assert pure_run["final_test_acc"] == 1.0
    assert pure_run["grokked"] is True
    assert pure_run["config_seed"] == 42
    assert "history" not in pure_run  # Should not be in top level

def test_confidence_intervals():
    """Test standard normal CI calculation."""
    df = pd.DataFrame({
        "config_condition_name": ["A", "A", "A", "A", "B", "B"],
        "metric": [1.0, 1.1, 0.9, 1.0, 5.0, 5.0]
    })

    ci = compute_confidence_intervals(df, "metric", "config_condition_name")

    # A mean should be 1.0, B mean should be 5.0
    a_res = ci[ci["Condition"] == "A"].iloc[0]
    b_res = ci[ci["Condition"] == "B"].iloc[0]

    assert np.isclose(a_res["Mean"], 1.0)
    assert np.isclose(b_res["Mean"], 5.0)
    assert b_res["Std"] == 0.0  # constant
    assert b_res["CI_Margin"] == 0.0

def test_bootstrap_ci():
    """Test bootstrap CI produces reasonable bounds."""
    np.random.seed(42)
    data = np.random.normal(loc=10.0, scale=1.0, size=100)

    mean, lower, upper = bootstrap_ci(data, num_bootstrap=500, confidence=0.95)

    assert np.isclose(mean, np.mean(data), atol=0.1)
    assert lower < mean < upper
    assert lower > 9.0 and upper < 11.0

def test_mann_whitney_u():
    """Test Mann-Whitney U test function."""
    df = pd.DataFrame({
        "config_condition_name": ["pure"] * 5 + ["collapse"] * 5,
        "grokking_step": [1000, 1100, 1050, 1200, 950, 5000, 5000, 5000, 5000, 5000]
    })

    res = mann_whitney_u_test(df, "grokking_step", baseline_group="pure")

    assert len(res) == 1
    row = res.iloc[0]
    assert row["Baseline"] == "pure"
    assert row["Comparison"] == "collapse"
    assert bool(row["Significant_05"]) is True  # p-value should be small
    assert row["P_Value"] < 0.05

def test_correlations():
    """Test Pearson and Spearman correlations."""
    df = pd.DataFrame({
        "config_collapse_level": [0.0, 0.1, 0.2, 0.3, 0.4],
        "grokking_step": [1000, 2000, 3000, 4000, 5000]  # Perfect linear correlation
    })

    corrs = compute_correlations(df, "config_collapse_level", "grokking_step")

    assert np.isclose(corrs["pearson_r"], 1.0)
    assert np.isclose(corrs["spearman_r"], 1.0)
    assert corrs["pearson_p"] < 0.05
