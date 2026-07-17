import os
import yaml
import json
import pytest
from pathlib import Path

def test_scaling_dry_run(tmp_path):
    """Test scaling sweep in dry-run mode."""
    from experiments.scaling import run_scaling_sweep

    config = {
        "experiment_name": "scaling_test",
        "depths": [1],
        "widths": [64],
        "collapse_levels": [0.0],
        "prime": 11,
        "max_steps": 2,
        "eval_every": 1,
        "seeds": [42],
        "output_dir": str(tmp_path / "scaling")
    }

    config_path = tmp_path / "scaling.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_scaling_sweep(str(config_path))

    assert (tmp_path / "scaling" / "scaling_summary.json").exists()

def test_curriculum_dry_run(tmp_path):
    """Test curriculum learning in dry-run mode."""
    from experiments.curriculum import run_curriculum_experiment

    config = {
        "experiment_name": "curriculum_test",
        "schedules": ["linear", "constant"],
        "start_collapse_level": 0.0,
        "end_collapse_level": 0.5,
        "step_transition_frac": 0.5,
        "prime": 11,
        "max_steps": 2,
        "eval_every": 1,
        "seeds": [42],
        "output_dir": str(tmp_path / "curriculum")
    }

    config_path = tmp_path / "curriculum.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_curriculum_experiment(str(config_path))

    assert (tmp_path / "curriculum" / "curriculum_summary.json").exists()

def test_threshold_dry_run(tmp_path):
    """Test threshold binary search in dry-run mode."""
    from experiments.threshold import run_threshold_experiment

    # Large tolerance so it ends in 1 iteration
    config = {
        "experiment_name": "threshold_test",
        "min_collapse": 0.0,
        "max_collapse": 0.5,
        "tolerance": 0.3,
        "prime": 11,
        "max_steps": 2,
        "eval_every": 1,
        "seeds": [42],
        "output_dir": str(tmp_path / "threshold")
    }

    config_path = tmp_path / "threshold.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_threshold_experiment(str(config_path))

    assert (tmp_path / "threshold" / "thresholds_summary.json").exists()

def test_extended_analysis(tmp_path, capsys):
    """Test that the analysis scripts can run over empty/dummy data without crashing."""
    from analysis.extended import run_all_analysis

    # Create empty mock directories
    (tmp_path / "scaling").mkdir()
    (tmp_path / "curriculum").mkdir()
    (tmp_path / "threshold").mkdir()

    # Create dummy summaries
    with open(tmp_path / "scaling" / "scaling_summary.json", "w") as f:
        json.dump([
            {"depth": 1, "width": 64, "collapse_level": 0.0, "seed": 42, "grokked": True, "grokking_step": 100}
        ], f)

    with open(tmp_path / "curriculum" / "curriculum_summary.json", "w") as f:
        json.dump([
            {"schedule": "linear", "seed": 42, "grokked": True, "grokking_step": 100}
        ], f)

    with open(tmp_path / "threshold" / "thresholds_summary.json", "w") as f:
        json.dump({
            "thresholds": [{"seed": 42, "threshold": 0.25}],
            "history": []
        }, f)

    run_all_analysis(str(tmp_path))

    captured = capsys.readouterr()
    assert "Scaling Analysis" in captured.out
    assert "Curriculum Analysis" in captured.out
    assert "Threshold Analysis" in captured.out
