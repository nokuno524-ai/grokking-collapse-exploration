import pytest
import numpy as np
import pandas as pd
import os

from src.analysis.comparison import ComparisonFramework

def test_measure_time_to_grokking():
    history = [
        {"step": 100, "test_acc": 0.1},
        {"step": 200, "test_acc": 0.5},
        {"step": 300, "test_acc": 0.95},
        {"step": 400, "test_acc": 0.96},
        {"step": 500, "test_acc": 0.94},
        {"step": 600, "test_acc": 0.97},
    ]

    # Needs 3 consecutive steps above 0.9
    step = ComparisonFramework.measure_time_to_grokking(history, threshold=0.9, consecutive_steps=3)
    assert step == 300

    # Needs 5 consecutive steps, only has 4, so it shouldn't find one
    step = ComparisonFramework.measure_time_to_grokking(history, threshold=0.9, consecutive_steps=5)
    assert step is None

def test_compute_auc():
    history = [
        {"step": 0, "test_acc": 0.0},
        {"step": 100, "test_acc": 0.5},
        {"step": 200, "test_acc": 1.0},
    ]

    auc = ComparisonFramework.compute_auc(history, metric="test_acc")
    # Area = triangle + rectangle = 0.5 * 100 * 0.5 + 100 * 0.5 + 0.5 * 100 * 0.5 = 25 + 50 + 25 = 100
    assert np.isclose(auc, 100.0)

def test_detect_phase_transitions():
    history = [
        {"step": 100, "test_acc": 0.1},
        {"step": 200, "test_acc": 0.15},
        {"step": 300, "test_acc": 0.2},
        {"step": 400, "test_acc": 0.8}, # jump of 0.6 over 100 steps -> deriv 0.006
        {"step": 500, "test_acc": 0.85},
    ]

    transitions = ComparisonFramework.detect_phase_transitions(history, metric="test_acc", threshold_derivative=0.005)
    assert len(transitions) == 1
    assert transitions[0] == 400

def test_correlate_collapse_with_grokking():
    df = pd.DataFrame({
        "collapse_severity": [0.0, 0.2, 0.5, 0.8],
        "grokking_step": [1000, 2000, 5000, 10000],
        "final_weight_norm": [30.0, 25.0, 20.0, 15.0]
    })

    corr = ComparisonFramework.correlate_collapse_with_grokking(df)
    assert "severity_vs_grokking_step" in corr
    assert "weight_norm_vs_grokking_step" in corr

    # As severity goes up, grokking step goes up -> positive correlation
    assert corr["severity_vs_grokking_step"] > 0
    # As weight norm goes down, grokking step goes up -> negative correlation
    assert corr["weight_norm_vs_grokking_step"] < 0

def test_generate_comparison_plots(tmp_path):
    results = {
        "pure": [
            {"final_test_acc": 1.0, "grokking_step": 1400, "history": [{"step": 100, "test_acc": 0.1}, {"step": 200, "test_acc": 0.9}]},
            {"final_test_acc": 1.0, "grokking_step": 1500, "history": [{"step": 100, "test_acc": 0.1}, {"step": 200, "test_acc": 0.95}]}
        ],
        "collapse": [
            {"final_test_acc": 0.2, "grokking_step": None, "history": [{"step": 100, "test_acc": 0.1}, {"step": 200, "test_acc": 0.2}]},
        ]
    }

    output_path = tmp_path / "plot.png"
    ComparisonFramework.generate_comparison_plots(results, str(output_path))

    assert os.path.exists(output_path)
