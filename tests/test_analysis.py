import pytest
import numpy as np
import pandas as pd
import torch
import json
import tempfile
from pathlib import Path

from src.analysis.parse_results import parse_results_json, aggregate_results
from src.analysis.statistics import bootstrap_confidence_interval, compute_correlation_matrix
from src.analysis.circuits import get_logit_attribution, activation_patching, integrated_gradients_attention
from src.model import ModularArithmeticTransformer

def test_parse_results_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data = {"config": {"seed": 42}, "grokked": True}
        file_path = tmpdir / "results.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        res = parse_results_json(file_path)
        assert res["grokked"] is True
        assert res["config"]["seed"] == 42

def test_aggregate_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cond_dir = tmpdir / "pure" / "seed_42"
        cond_dir.mkdir(parents=True)
        data = {
            "config": {"seed": 42, "collapse_level": 0.0, "condition_name": "pure"},
            "grokked": True,
            "grokking_step": 1400
        }
        with open(cond_dir / "results.json", "w") as f:
            json.dump(data, f)

        df = aggregate_results(results_dir=str(tmpdir))
        assert len(df) == 1
        assert df.iloc[0]["condition"] == "pure"
        assert bool(df.iloc[0]["grokked"]) is True
        assert df.iloc[0]["grokking_step"] == 1400

def test_bootstrap_confidence_interval():
    data = np.array([10, 10, 10, 10, 10])
    lower, upper = bootstrap_confidence_interval(data)
    assert np.isclose(lower, 10.0)
    assert np.isclose(upper, 10.0)

def test_compute_correlation_matrix():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [1.0, 2.0, 3.0],
        "c": [3.0, 2.0, 1.0]
    })
    corr = compute_correlation_matrix(df, ["a", "b", "c"])
    assert np.isclose(corr.loc["a", "b"], 1.0)
    assert np.isclose(corr.loc["a", "c"], -1.0)

def test_circuits_functions():
    model = ModularArithmeticTransformer(prime=5)
    model.eval()
    inputs = torch.tensor([[1, 2]])
    target_idx = 3

    attr = get_logit_attribution(model, inputs, target_idx)
    assert "embed_contrib" in attr
    assert "attn_contrib" in attr
    assert "mlp_contrib" in attr
    assert "total_logit" in attr

    corrupt_model = ModularArithmeticTransformer(prime=5)
    corrupt_model.eval()
    patch_out = activation_patching(model, corrupt_model, inputs, patch_layer='embed')
    assert patch_out.shape == (1, 5)

    ig = integrated_gradients_attention(model, inputs, target_idx, steps=5)
    assert ig.shape == (model.d_model,)
