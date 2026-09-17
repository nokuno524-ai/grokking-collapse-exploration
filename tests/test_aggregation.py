import json
from pathlib import Path
from src.analysis.grok_detector.run_aggregator import group_runs_by_severity
from src.analysis.grok_detector.stats import compute_kaplan_meier
import numpy as np

def test_group_runs_by_severity(tmp_path):
    res_dir = tmp_path / "results"
    res_dir.mkdir()

    r1_dir = res_dir / "run1"
    r1_dir.mkdir()
    with open(r1_dir / "results.json", "w") as f:
        json.dump({"config": {"collapse_severity": 0.0}, "grokked": True}, f)

    r2_dir = res_dir / "run2"
    r2_dir.mkdir()
    with open(r2_dir / "results.json", "w") as f:
        json.dump({"config": {"collapse_severity": 0.5}, "grokked": False}, f)

    r3_dir = res_dir / "run3"
    r3_dir.mkdir()
    with open(r3_dir / "results.json", "w") as f:
        json.dump({"config": {"collapse_severity": 0.5}, "grokked": True}, f)

    grouped = group_runs_by_severity(res_dir)
    assert 0.0 in grouped
    assert len(grouped[0.0]) == 1
    assert 0.5 in grouped
    assert len(grouped[0.5]) == 2

def test_censoring_correctness():
    runs = [
        {"grokked": True, "grokking_step": 100},
        {"grokked": True, "grokking_step": 150},
        {"grokked": False}, # censored
        {"grokked": False}  # censored
    ]
    max_steps = 200
    times, probs = compute_kaplan_meier(runs, max_steps)

    assert len(times) == 4 # 0, 100, 150, 200
    assert list(times) == [0, 100, 150, 200]

    # At t=0, prob = 1.0
    # At t=100, 1 event out of 4 -> prob = 1.0 * (3/4) = 0.75
    # At t=150, 1 event out of 3 -> prob = 0.75 * (2/3) = 0.5
    # At t=200, 0 events out of 2 (censored) -> prob = 0.5
    assert np.allclose(probs, [1.0, 0.75, 0.5, 0.5])
