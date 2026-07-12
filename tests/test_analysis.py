import pytest
import numpy as np
import os
import json
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

# Add project root to sys.path if not running from root directly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.analyze_results import analyze_results
from analysis.statistics import bootstrap_ci, load_final_metrics
from analysis.visualizations import load_histories

@pytest.fixture
def dummy_results_dir():
    with TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)

        # Create a pure condition
        pure = base / "pure"
        pure.mkdir()
        pure_data = {
            "config": {"condition_name": "pure", "collapse_level": 0.0, "collapse_severity": 0.5},
            "grokked": True,
            "final_test_acc": 1.0,
            "history": [
                {"step": 100, "test_acc": 0.5, "test_loss": 2.0},
                {"step": 200, "test_acc": 1.0, "test_loss": 0.1}
            ]
        }
        with open(pure / "results.json", "w") as f:
            json.dump(pure_data, f)

        # Create a severe_collapse condition
        severe = base / "severe_collapse"
        severe.mkdir()
        severe_data = {
            "config": {"condition_name": "severe_collapse", "collapse_level": 0.5, "collapse_severity": 0.9},
            "grokked": False,
            "final_test_acc": 0.1,
            "history": [
                {"step": 100, "test_acc": 0.2, "test_loss": 4.0},
                {"step": 200, "test_acc": 0.1, "test_loss": 5.0}
            ]
        }
        with open(severe / "results.json", "w") as f:
            json.dump(severe_data, f)

        yield base

def test_bootstrap_ci():
    data = np.random.normal(0, 1, 1000)
    mean, lower, upper = bootstrap_ci(data, num_samples=100)
    assert lower <= mean <= upper
    assert len(data) == 1000

def test_load_final_metrics(dummy_results_dir):
    metrics = load_final_metrics(results_dir=dummy_results_dir)
    assert len(metrics) == 2
    conds = [m["condition"] for m in metrics]
    assert "pure" in conds
    assert "severe_collapse" in conds

def test_load_histories(dummy_results_dir):
    histories = load_histories(results_dir=dummy_results_dir)
    assert len(histories) == 2
    assert "pure" in histories
    assert len(histories["pure"]) == 2
    assert histories["pure"][1]["test_acc"] == 1.0

def test_analyze_results(dummy_results_dir, capsys):
    analyze_results(results_dir=dummy_results_dir)
    # Check if CSV/MD were created
    assert (dummy_results_dir / "comprehensive_summary.csv").exists()
    assert (dummy_results_dir / "ANALYSIS.md").exists()

    # Check stdout
    captured = capsys.readouterr()
    assert "Experiment Results Summary" in captured.out
    assert "pure" in captured.out
    assert "severe_collapse" in captured.out
