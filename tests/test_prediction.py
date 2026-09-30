import pytest
import numpy as np
import tempfile
import torch
import json
from pathlib import Path

from src.prediction.dose_response_harness import generate_configs
from src.prediction.early_warning import (
    compute_effective_rank,
    compute_effective_rank_from_checkpoint,
    compute_weight_norm_trajectory,
    compute_train_test_gap_derivative,
    compute_attention_entropy_slope
)
from src.prediction.outcome_predictor import extract_features

def test_dose_response_harness_configs():
    configs = generate_configs([0.0, 0.1], [0.2, 0.4], [42], "test_out")
    assert len(configs) == 4
    c = configs[1]
    assert c.collapse_level == 0.0
    assert c.collapse_severity == 0.4
    assert c.seed == 42
    assert "level0_sev0.4" in c.output_dir

def test_compute_effective_rank():
    # uniform
    rank_u = compute_effective_rank(np.ones(10))
    # near singular
    s = np.zeros(10)
    s[0] = 1.0
    rank_s = compute_effective_rank(s)

    assert rank_u > 9.0 # ~10
    assert rank_s < 1.1 # ~1
    assert np.isnan(compute_effective_rank([]))

def test_compute_weight_norm_trajectory():
    history = [{"weight_norm": 10.0}, {"weight_norm": 5.0}, {}]
    res = compute_weight_norm_trajectory(history)
    assert res[:2] == [10.0, 5.0]
    assert np.isnan(res[2])

def test_compute_train_test_gap_derivative():
    history = [
        {"step": 0, "train_loss": 1.0, "test_loss": 1.0},
        {"step": 1000, "train_loss": 0.5, "test_loss": 1.5},
        {"step": 2000, "train_loss": 0.1, "test_loss": 2.1}
    ]
    # gap goes: 0, 1.0, 2.0. Change is +1.0 per 1000 steps.
    slope = compute_train_test_gap_derivative(history, window=3)
    assert pytest.approx(slope) == 1.0

    # short window
    assert compute_train_test_gap_derivative(history, window=10) is None

def test_extract_features(tmp_path):
    run_dir = tmp_path / "seed_42"
    run_dir.mkdir()

    history = [
        {"step": 1000, "train_loss": 1.0, "test_loss": 1.0, "weight_norm": 5.0},
        {"step": 2000, "train_loss": 0.5, "test_loss": 1.5, "weight_norm": 4.0},
        {"step": 3000, "train_loss": 0.1, "test_loss": 2.1, "weight_norm": 3.0},
        {"step": 4000, "test_acc": 1.0} # grokking step
    ]
    data = {
        "history": history,
        "config": {"collapse_level": 0.1, "collapse_severity": 0.5},
        "grokked": True,
        "grokking_step": 4000
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(data, f)

    feats = extract_features(str(run_dir))
    assert feats["grokked"] is True
    assert feats["is_collapsed"] is True
    # At 25% of grokking step (1000), there is only 1 history entry, so slope is nan
    assert np.isnan(feats["tt_gap_slope"])
    assert np.isnan(feats["eff_rank_last"])
    assert np.isnan(feats["attn_entropy_slope"])
