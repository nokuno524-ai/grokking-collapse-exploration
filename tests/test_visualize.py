import os
import json
import pytest
import sys

# Ensure tools can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.visualize_results import detect_grokking_point, analyze_results

def test_detect_grokking_point():
    """Test detect_grokking_point correctly identifies grokking points and edge cases."""

    # 1. Grokking occurs and stays above threshold for 50 steps
    # Here, steps increment by 10. We need length equivalent to 50 steps.
    # From step 60 (idx 5) to 110 (idx 10) it stays above threshold 0.9.
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    accuracies = [0.1, 0.2, 0.3, 0.8, 0.85, 0.91, 0.95, 0.92, 0.98, 0.99, 1.0, 1.0]

    assert detect_grokking_point(steps, accuracies, threshold=0.9) == 60

    # 2. Never hits threshold
    accuracies_low = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.89, 0.88, 0.89, 0.89]
    assert detect_grokking_point(steps, accuracies_low, threshold=0.9) is None

    # 3. Hits threshold but dips below it within 50 steps
    # At step 60, acc = 0.91, but at step 80 (within 50 steps of 60), acc = 0.85
    # At step 60 (idx 5), acc = 0.91, but at step 80 (idx 7, step 60+20), acc = 0.85
    # Then at step 90 (idx 8) it hits 0.98, and from step 90 to 140 it stays above.
    steps_dip = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    accuracies_dip = [0.1, 0.2, 0.3, 0.8, 0.85, 0.91, 0.95, 0.85, 0.98, 0.92, 0.95, 0.99, 1.0, 1.0, 1.0, 1.0]
    assert detect_grokking_point(steps_dip, accuracies_dip, threshold=0.9) == 90


def test_analyze_results(tmpdir):
    """Test analyze_results correctly aggregates data from multiple JSON files."""

    # Setup mock directory structure and JSONs
    results_dir = tmpdir.mkdir("results")

    cond1_dir = results_dir.mkdir("pure")
    cond1_data = {
        "final_test_acc": 1.0,
        "history": [
            {"step": 100, "weight_norm": 50.0, "test_acc": 0.1},
            {"step": 200, "weight_norm": 45.0, "test_acc": 0.95},
            {"step": 300, "weight_norm": 30.0, "test_acc": 1.0}
        ]
    }
    with open(cond1_dir.join("results.json"), "w") as f:
        json.dump(cond1_data, f)

    cond2_dir = results_dir.mkdir("collapse")
    cond2_data = {
        "final_test_acc": 0.5,
        "history": [
            {"step": 100, "weight_norm": 50.0, "test_acc": 0.1},
            {"step": 200, "weight_norm": 48.0, "test_acc": 0.3},
            {"step": 300, "weight_norm": 45.0, "test_acc": 0.5}
        ]
    }
    with open(cond2_dir.join("results.json"), "w") as f:
        json.dump(cond2_data, f)

    stats = analyze_results(str(results_dir))

    assert "pure" in stats
    assert "collapse" in stats

    assert stats["pure"]["final_test_acc"] == 1.0
    assert stats["pure"]["grokking_point"] == 200
    assert stats["pure"]["weight_norm_reduction"] == 20.0

    assert stats["collapse"]["final_test_acc"] == 0.5
    assert stats["collapse"]["grokking_point"] is None
    assert stats["collapse"]["weight_norm_reduction"] == 5.0
