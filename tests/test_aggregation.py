"""
Unit tests for the aggregation and stats modules.
"""

import numpy as np
import pytest

from src.rigorous_stats.stats_module import compute_bootstrap_ci, run_mannwhitneyu
from src.ensemble.aggregation import aggregate_metrics

def test_bootstrap_ci_reproducibility():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    mean1, low1, high1 = compute_bootstrap_ci(data, num_bootstraps=100, seed=42)
    mean2, low2, high2 = compute_bootstrap_ci(data, num_bootstraps=100, seed=42)

    assert mean1 == mean2
    assert low1 == low2
    assert high1 == high2
    assert mean1 == 3.0

def test_bootstrap_ci_ragged_data():
    data = [1.0, np.nan, 3.0, None, 5.0]
    mean, low, high = compute_bootstrap_ci(data, num_bootstraps=100, seed=42)

    # 1, 3, 5 -> mean 3
    assert mean == 3.0
    assert low <= mean <= high

def test_mannwhitneyu_basic():
    group1 = [1.0, 1.1, 1.2, 1.0, 1.1]
    group2 = [5.0, 5.1, 5.2, 5.0, 5.1]

    res = run_mannwhitneyu(group1, group2)
    assert res['pvalue'] < 0.05
    assert res['statistic'] == 0.0  # group 1 is completely below group 2

def test_mannwhitneyu_ragged():
    group1 = [1.0, np.nan, 1.2, 1.0, 1.1]
    group2 = [5.0, 5.1, None, 5.0, 5.1]

    res = run_mannwhitneyu(group1, group2)
    assert res['pvalue'] < 0.05

def test_mannwhitneyu_identical():
    group1 = [1.0, 1.0, 1.0]
    group2 = [1.0, 1.0, 1.0]

    res = run_mannwhitneyu(group1, group2)
    assert np.isnan(res['pvalue']) or res['pvalue'] > 0.05

def test_aggregate_metrics():
    results = {
        "pure": [
            {"final_test_acc": 1.0, "grokking_step": 1000, "final_fourier_concentration": 0.9},
            {"final_test_acc": 0.98, "grokking_step": 1200, "final_fourier_concentration": 0.88},
            {"final_test_acc": 1.0, "grokking_step": 900, "final_fourier_concentration": 0.92},
        ],
        "noisy": [
            {"final_test_acc": 0.1, "final_fourier_concentration": 0.1},
            {"final_test_acc": 0.1, "final_fourier_concentration": 0.1},
        ]
    }

    df = aggregate_metrics(results)
    assert len(df) == 2
    assert "Condition" in df.columns
    assert "Test Acc (Mean ± Std)" in df.columns

    pure_row = df[df["Condition"] == "pure"].iloc[0]
    # Check that Mean ± Std format string was generated correctly
    assert "1.0000" in pure_row["Test Acc (Mean ± Std)"] or "0.9933" in pure_row["Test Acc (Mean ± Std)"]

    noisy_row = df[df["Condition"] == "noisy"].iloc[0]
    assert "0.1000" in noisy_row["Test Acc (Mean ± Std)"]
    assert noisy_row["Grok Step (Mean ± Std)"] == "NaN"

def test_aggregate_trajectories():
    from src.ensemble.aggregation import aggregate_trajectories
    results = {
        "pure": [
            {
                "history": [
                    {"step": 10, "weight_norm": 1.0},
                    {"step": 20, "weight_norm": 2.0},
                    {"step": 30, "weight_norm": 3.0}
                ]
            },
            {
                "history": [
                    {"step": 10, "weight_norm": 1.2},
                    # run 2 skipped step 20
                    {"step": 30, "weight_norm": 3.2},
                    {"step": 40, "weight_norm": 4.0}
                ]
            }
        ]
    }

    agg = aggregate_trajectories(results, "weight_norm")

    assert "pure" in agg
    pure_agg = agg["pure"]

    # Check that all unique steps were captured
    assert pure_agg["steps"] == [10, 20, 30, 40]

    # Step 10 has 2 values (1.0, 1.2) -> mean 1.1
    assert np.isclose(pure_agg["mean"][0], 1.1)

    # Step 20 has 1 value (2.0)
    assert np.isclose(pure_agg["mean"][1], 2.0)
    assert pure_agg["std"][1] == 0.0 # 1 element -> std is 0

    # Step 40 has 1 value (4.0)
    assert np.isclose(pure_agg["mean"][3], 4.0)
