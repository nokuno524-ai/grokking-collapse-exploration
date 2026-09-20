import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import math
from pathlib import Path
import json
import shutil
import tempfile

from src.ensemble.runner import build_config_matrix, is_task_completed
from src.ensemble.aggregation import compute_kaplan_meier, estimate_median_survival, aggregate_results

def test_config_matrix_expansion():
    """Test that the configuration matrix is expanded correctly."""
    seeds = [1, 2]
    conditions = ["pure", "low_collapse"]

    tasks = build_config_matrix(seeds, conditions)

    assert len(tasks) == 4

    # Check all combinations are present
    combinations = set((t["seed"], t["condition_name"]) for t in tasks)
    expected = {(1, "pure"), (1, "low_collapse"), (2, "pure"), (2, "low_collapse")}
    assert combinations == expected

    # Check that configs have the right properties
    for t in tasks:
        # assert t["config"].seed == t["seed"] # Base seed is not assigned directly to config in runner
        assert "collapse_level" in dir(t["config"])

def test_is_task_completed():
    """Test resumability check."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test false when not exists
        assert not is_task_completed(temp_path)

        # Test true when exists
        (temp_path / "results.json").touch()
        assert is_task_completed(temp_path)

def test_kaplan_meier_simple():
    """Test Kaplan-Meier calculation with known synthetic data."""
    # Format: (time, event_occurred)
    # 2 grok at 100, 1 grok at 200, 1 censored at 200, 1 grok at 300
    events = [
        (100, True),
        (100, True),
        (200, True),
        (200, False), # Censored
        (300, True)
    ]

    times, probs, variances = compute_kaplan_meier(events, max_time=500)

    # t=0: all 5 at risk, prob=1.0
    # t=100: 5 at risk, 2 grok. prob_surv = 1.0 * (1 - 2/5) = 0.6
    # t=200: 3 at risk, 1 grok, 1 censor. prob_surv = 0.6 * (1 - 1/3) = 0.4
    # t=300: 1 at risk, 1 grok. prob_surv = 0.4 * (1 - 1/1) = 0.0

    assert times[0] == 0
    assert times[1] == 100
    assert times[2] == 200
    assert times[3] == 300
    assert times[4] == 500 # extended to max_time

    assert math.isclose(probs[0], 1.0)
    assert math.isclose(probs[1], 0.6)
    assert math.isclose(probs[2], 0.4)
    assert math.isclose(probs[3], 0.0)
    assert math.isclose(probs[4], 0.0)

    median = estimate_median_survival(times, probs)
    # Falls below 0.5 at t=200
    assert median == 200

def test_aggregate_results():
    """Test aggregation end-to-end on synthetic files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Create some mock results

        # Run 1: grokked at 150
        dir1 = base_dir / "1" / "pure"
        dir1.mkdir(parents=True)
        with open(dir1 / "results.json", "w") as f:
            json.dump({
                "config": {"condition_name": "pure", "seed": 1},
                "grokked": True,
                "grokking_step": 150,
                "final_test_acc": 0.98
            }, f)

        # Run 2: grokked at 250
        dir2 = base_dir / "2" / "pure"
        dir2.mkdir(parents=True)
        with open(dir2 / "results.json", "w") as f:
            json.dump({
                "config": {"condition_name": "pure", "seed": 2},
                "grokked": True,
                "grokking_step": 250,
                "final_test_acc": 0.99
            }, f)

        # Run 3: never grokked
        dir3 = base_dir / "3" / "pure"
        dir3.mkdir(parents=True)
        with open(dir3 / "results.json", "w") as f:
            json.dump({
                "config": {"condition_name": "pure", "seed": 3},
                "grokked": False,
                "grokking_step": None,
                "final_test_acc": 0.40
            }, f)

        aggregated = aggregate_results(str(base_dir), max_steps=500)

        assert "pure" in aggregated
        stats = aggregated["pure"]

        assert stats["n_runs"] == 3
        assert stats["n_grokked"] == 2
        assert stats["median_grok_step"] == 250

        # Times should be 0, 150, 250, 500
        assert stats["km_times"] == [0, 150, 250, 500]

        # t=0: 3 at risk, S=1.0
        # t=150: 3 at risk, 1 grok, S = 1 * 2/3 = 0.666...
        # t=250: 2 at risk, 1 grok, S = 0.666... * 1/2 = 0.333...
        # t=500: 1 at risk, 0 grok (censored), S = 0.333...
        assert math.isclose(stats["km_probs"][1], 2/3)
        assert math.isclose(stats["km_probs"][2], 1/3)
        assert math.isclose(stats["km_probs"][3], 1/3)
