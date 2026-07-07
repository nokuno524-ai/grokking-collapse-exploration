import pytest
import os
import json
from pathlib import Path
from run_experiments import check_run_completed

def test_check_run_completed(tmp_path):
    output_dir = tmp_path / "results"
    condition_dir = output_dir / "pure"
    condition_dir.mkdir(parents=True)

    # Test 1: No results file
    assert not check_run_completed(str(output_dir), "pure")

    # Test 2: Incomplete results file (missing "grokked" key)
    results_file = condition_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump({"step": 100}, f)

    assert not check_run_completed(str(output_dir), "pure")

    # Test 3: Complete results file
    with open(results_file, "w") as f:
        json.dump({"grokked": True, "grokking_step": 1000}, f)

    assert check_run_completed(str(output_dir), "pure")
