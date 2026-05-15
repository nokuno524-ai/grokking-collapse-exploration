import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.analysis.results import ExperimentAnalyzer

@pytest.fixture
def mock_results_dir(tmp_path):
    """Create a mock results directory with synthetic data."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Condition 1: Pure
    pure_dir = results_dir / "pure"
    pure_dir.mkdir()

    pure_data1 = {
        "config": {"condition_name": "pure"},
        "grokking_step": 1400,
        "final_train_acc": 1.0,
        "final_test_acc": 1.0,
        "final_weight_norm": 30.0,
    }
    pure_data2 = {
        "config": {"condition_name": "pure"},
        "grokking_step": 1500,
        "final_train_acc": 1.0,
        "final_test_acc": 1.0,
        "final_weight_norm": 28.0,
    }

    (pure_dir / "seed_1").mkdir()
    with open(pure_dir / "seed_1" / "results.json", "w") as f:
        json.dump(pure_data1, f)

    (pure_dir / "seed_2").mkdir()
    with open(pure_dir / "seed_2" / "results.json", "w") as f:
        json.dump(pure_data2, f)

    # Condition 2: Collapse
    collapse_dir = results_dir / "collapse"
    collapse_dir.mkdir()

    col_data1 = {
        "config": {"condition_name": "collapse"},
        "grokking_step": None,
        "final_train_acc": 0.8,
        "final_test_acc": 0.2,
        "final_weight_norm": 20.0,
    }
    col_data2 = {
        "config": {"condition_name": "collapse"},
        "grokking_step": None,
        "final_train_acc": 0.85,
        "final_test_acc": 0.25,
        "final_weight_norm": 18.0,
    }

    (collapse_dir / "seed_1").mkdir()
    with open(collapse_dir / "seed_1" / "results.json", "w") as f:
        json.dump(col_data1, f)

    (collapse_dir / "seed_2").mkdir()
    with open(collapse_dir / "seed_2" / "results.json", "w") as f:
        json.dump(col_data2, f)

    return results_dir


def test_load_results(mock_results_dir):
    analyzer = ExperimentAnalyzer()
    analyzer.load_results(mock_results_dir)

    assert "pure" in analyzer.results
    assert "collapse" in analyzer.results
    assert len(analyzer.results["pure"]) == 2
    assert len(analyzer.results["collapse"]) == 2


def test_compute_summary_statistics(mock_results_dir):
    analyzer = ExperimentAnalyzer()
    analyzer.load_results(mock_results_dir)

    df = analyzer.compute_summary_statistics(metrics=["final_test_acc", "final_weight_norm"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    pure_row = df[df["condition"] == "pure"].iloc[0]
    assert pure_row["final_test_acc_mean"] == 1.0
    assert pure_row["final_weight_norm_mean"] == 29.0

    col_row = df[df["condition"] == "collapse"].iloc[0]
    assert col_row["final_test_acc_mean"] == 0.225
    assert col_row["final_weight_norm_mean"] == 19.0


def test_statistical_significance(mock_results_dir):
    analyzer = ExperimentAnalyzer()
    analyzer.load_results(mock_results_dir)

    # Force more data to make bootstrap CI stable
    analyzer.results["pure"] *= 5
    analyzer.results["collapse"] *= 5

    results = analyzer.test_statistical_significance("pure", "collapse", "final_weight_norm", bootstrap_samples=100)

    assert "ttest" in results
    assert "mann_whitney" in results
    assert "bootstrap_ci" in results

    # Means are 29.0 and 19.0, so they should be significantly different
    assert results["ttest"]["p_value"] < 0.05
    assert results["bootstrap_ci"]["low"] > 0


def test_compute_effect_sizes(mock_results_dir):
    analyzer = ExperimentAnalyzer()
    analyzer.load_results(mock_results_dir)

    d = analyzer.compute_effect_sizes("pure", "collapse", "final_weight_norm")
    # Mean1 = 29, Var1 = 2, Mean2 = 19, Var2 = 2. Pooled var = 2. SD = sqrt(2). d = 10 / sqrt(2) = 7.07
    assert np.isclose(d, 7.071, atol=0.01)


def test_generate_latex_table(mock_results_dir):
    analyzer = ExperimentAnalyzer()
    analyzer.load_results(mock_results_dir)

    latex = analyzer.generate_latex_table(metrics=["final_test_acc"])
    assert "\\begin{table}" in latex
    assert "pure" in latex
    assert "collapse" in latex
    assert "1.000" in latex
