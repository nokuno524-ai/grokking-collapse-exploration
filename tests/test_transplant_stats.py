import numpy as np
import pytest
from src.transplant.stats import cohens_d, bootstrap_ci, check_replication

def test_cohens_d_independent():
    g1 = [1, 2, 3, 4, 5]
    g2 = [3, 4, 5, 6, 7] # mean diff is 2, var is 2.5 for both
    # pooled_var = ((4 * 2.5) + (4 * 2.5)) / 8 = 2.5
    # d = (5 - 3) / sqrt(2.5) = 2 / 1.581 = 1.2649
    d = cohens_d(g1, g2, paired=False)
    assert np.isclose(d, 1.264911)

def test_cohens_d_paired():
    g1 = [1, 2, 3, 4, 5]
    g2 = [3, 4, 5, 6, 7]
    # diffs = [2, 2, 2, 2, 2]
    # mean diff = 2, var diff = 0 -> d = 0.0
    d = cohens_d(g1, g2, paired=True)
    assert d == 0.0

    g2 = [2, 4, 4, 6, 6]
    # diffs = [1, 2, 1, 2, 1]
    # mean diff = 1.4, var diff = 0.3
    # d = 1.4 / sqrt(0.3) = 1.4 / 0.5477 = 2.556
    d = cohens_d(g1, g2, paired=True)
    assert np.isclose(d, 2.556038)

def test_cohens_d_edge_cases():
    assert cohens_d([], []) == 0.0
    assert cohens_d([1], [2], paired=False) == 0.0
    assert cohens_d([1], [2], paired=True) == 0.0
    assert cohens_d([1, 1, 1], [1, 1, 1], paired=False) == 0.0

def test_bootstrap_ci():
    data = [1, 2, 3, 4, 5]
    lower, upper = bootstrap_ci(data, seed=42)
    assert lower <= 3.0 <= upper
    assert lower >= 1.0
    assert upper <= 5.0

def test_bootstrap_ci_edge_cases():
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([5]) == (5.0, 5.0)

def test_check_replication():
    assert check_replication([1.0, 2.0, 0.5, 3.0]) == True
    assert check_replication([-1.0, -2.0, -0.5, -3.0]) == True
    assert check_replication([1.0, -1.0, 2.0]) == False
    assert check_replication([0.0, 0.0, 0.0]) == False
    assert check_replication([1.0, 2.0, 0.0]) == True # ignores zeros if there's signal
    assert check_replication([]) == False

import json
from pathlib import Path
from src.transplant.replication_harness import run_replication

def test_replication_harness_smoke(tmp_path):
    # Setup mock data directory structure
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "output"

    for seed in [42, 43]:
        for cond in ["pure", "low_collapse"]:
            run_dir = results_dir / str(seed) / cond
            run_dir.mkdir(parents=True, exist_ok=True)
            # Create a mock checkpoint and results.json (but wait, run_transplant_experiment will try to actually load torch models.
            # To just test the script logic without running real models, we might need a mock or small model, but since we are just doing a smoke test, maybe it's fine if it skips because it can't load, or we should patch run_transplant_experiment)

    # Let's just patch run_transplant_experiment for the smoke test to avoid needing real models
    from unittest.mock import patch
    from src.transplant.run_transplants import VariantResult

    with patch("src.transplant.replication_harness.run_transplant_experiment") as mock_run:
        mock_run.return_value = [
            VariantResult(name="baseline_contam", component=None, test_loss=1.0, test_acc=0.5, train_loss=1.0, train_acc=0.5, fourier_concentration=0.1, weight_norm=1.0),
            VariantResult(name="transplant_token_embed", component="token_embed", test_loss=0.5, test_acc=0.8, train_loss=0.5, train_acc=0.8, fourier_concentration=0.5, weight_norm=1.0)
        ]

        run_replication(
            seeds=[42, 43],
            conditions=["low_collapse"],
            results_dir=results_dir,
            output_dir=output_dir,
            components=["token_embed"],
            rescue_steps=0
        )

        assert (output_dir / "replication_all_results.json").exists()
        assert (output_dir / "replication_stats.json").exists()
        assert (output_dir / "replication_summary.md").exists()

        with open(output_dir / "replication_stats.json") as f:
            stats = json.load(f)

        assert len(stats) == 1
        assert stats[0]["condition"] == "low_collapse"
        assert stats[0]["mean_baseline_acc"] == 0.5
        assert stats[0]["mean_transplant_acc"] == 0.8
