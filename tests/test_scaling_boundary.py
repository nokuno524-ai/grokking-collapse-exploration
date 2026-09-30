import pytest
from pathlib import Path
import json
import tempfile
import math
from src.analysis.scaling_boundary import parse_scaling_results, aggregate_boundary, generate_phase_diagram, fit_scaling_law

def test_parse_scaling_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        d1 = tmp_path / "run1"
        d1.mkdir()
        with open(d1 / "results.json", "w") as f:
            json.dump({
                "config": {"d_model": 64, "collapse_level": 0.1, "seed": 1},
                "final_test_acc": 0.995,
                "grokking_step": 5000
            }, f)

        rows = parse_scaling_results(tmp_path)
        assert len(rows) == 1
        assert rows[0]["width"] == 64.0
        assert rows[0]["fraction"] == 0.1
        assert rows[0]["seed"] == 1
        assert rows[0]["final_test_acc"] == 0.995
        assert rows[0]["grokking_step"] == 5000

def test_aggregate_boundary():
    # Width 64, seed 1
    # Fra: 0.1 (G), 0.2 (G), 0.3 (F), 0.4 (F) -> boundary is 0.3
    rows = [
        {"width": 64, "dataset_size": 1.0, "seed": 1, "fraction": 0.1, "final_test_acc": 0.995},
        {"width": 64, "dataset_size": 1.0, "seed": 1, "fraction": 0.2, "final_test_acc": 0.991},
        {"width": 64, "dataset_size": 1.0, "seed": 1, "fraction": 0.3, "final_test_acc": 0.5},
        {"width": 64, "dataset_size": 1.0, "seed": 1, "fraction": 0.4, "final_test_acc": 0.2},
    ]
    bound = aggregate_boundary(rows)
    assert bound[(64.0, 1.0, 1)] == 0.3

    # All grok -> right censored (max + 0.05)
    rows2 = [
        {"width": 128, "dataset_size": 1.0, "seed": 1, "fraction": 0.1, "final_test_acc": 0.995},
        {"width": 128, "dataset_size": 1.0, "seed": 1, "fraction": 0.2, "final_test_acc": 0.991},
    ]
    bound2 = aggregate_boundary(rows2)
    assert bound2[(128.0, 1.0, 1)] == 0.25

    # None grok -> left censored (0)
    rows3 = [
        {"width": 256, "dataset_size": 1.0, "seed": 1, "fraction": 0.1, "final_test_acc": 0.2},
        {"width": 256, "dataset_size": 1.0, "seed": 1, "fraction": 0.2, "final_test_acc": 0.1},
    ]
    bound3 = aggregate_boundary(rows3)
    assert bound3[(256.0, 1.0, 1)] == 0.0

def test_fit_scaling_law_handling_nans():
    # Test that fit_scaling_law gracefully handles the case where fitting fails or returns NaN
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Supply a single point, fitting will fail and return NaN
        bound = {(64.0, 1.0, 1): 0.2}
        fit_scaling_law(bound, [], tmp_path)

        with open(tmp_path / "scaling_summary.md", "r") as f:
            content = f.read()
            assert "nan" in content or "NaN" in content
