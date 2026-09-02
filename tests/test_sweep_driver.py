import os
import json
from pathlib import Path
import pytest
from src.sweep.driver import build_tasks, emit_sbatch, run_one

def test_build_tasks():
    c_levels = [0.0, 0.1]
    t_fracs = [0.2, 0.5]
    d_mods = [32, 64]
    seeds = [42]

    tasks = build_tasks(c_levels, t_fracs, d_mods, seeds)

    assert len(tasks) == 8

    # Check if a specific combination exists
    assert (0.1, 0.5, 64, 42) in tasks

def test_emit_sbatch(tmp_path):
    tasks = [(0.0, 0.2, 32, 42), (0.1, 0.5, 64, 43)]

    # Temporarily change directory to write the file in a controlled location if needed,
    # or patch the `emit_sbatch` function's hardcoded path if necessary.
    # Since emit_sbatch currently writes to "slurm/phase_diagram.sbatch" in CWD,
    # we will mock it by changing CWD.

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        emit_sbatch(tasks, "dummy_out_dir", 50000)

        assert (tmp_path / "slurm" / "phase_diagram.sbatch").exists()

        with open(tmp_path / "slurm" / "phase_diagram.sbatch") as f:
            content = f.read()
            assert "--array=0-1%50" in content
            assert "dummy_out_dir" in content
    finally:
        os.chdir(orig_cwd)

def test_run_one_skip_logic(tmp_path):
    """Test that run_one correctly skips if results.json exists."""

    # Create the expected directory structure
    # d32/f0.2/c0/seed_42
    out_dir = tmp_path / "d32" / "f0.2" / "c0" / "seed_42"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy results.json
    results_file = out_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump({"dummy": "data"}, f)

    # Call run_one
    # If it skips, it returns None. If it doesn't skip, it would normally
    # invoke train() which returns a TrainState.
    result = run_one(0.0, 0.2, 32, 42, str(tmp_path), 10)

    assert result is None
