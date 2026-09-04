import pytest
import torch
import numpy as np

from src.transplant.head_transplant import patch_head, get_head_indices

def test_get_head_indices():
    sl = get_head_indices(128, 4, 0)
    assert sl.start == 0 and sl.stop == 32
    sl = get_head_indices(128, 4, 1)
    assert sl.start == 32 and sl.stop == 64
    sl = get_head_indices(128, 4, 3)
    assert sl.start == 96 and sl.stop == 128

def test_patch_head_swap():
    d_model = 128
    n_heads = 4
    d_head = d_model // n_heads

    base_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(3 * d_model, d_model),
        "transformer.layers.0.self_attn.in_proj_bias": torch.zeros(3 * d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(d_model, d_model),
        "transformer.layers.0.self_attn.out_proj.bias": torch.zeros(d_model)
    }

    donor_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(3 * d_model, d_model),
        "transformer.layers.0.self_attn.in_proj_bias": torch.ones(3 * d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.ones(d_model, d_model),
        "transformer.layers.0.self_attn.out_proj.bias": torch.ones(d_model)
    }

    out_sd = patch_head(base_sd, donor_sd, 0, 1, d_model, n_heads, mode="swap")

    # Check in_proj for Q, K, V
    for i in range(3):
        start = i * d_model
        # head 0 should be 0
        assert torch.all(out_sd["transformer.layers.0.self_attn.in_proj_weight"][start:start+d_head] == 0)
        # head 1 should be 1
        assert torch.all(out_sd["transformer.layers.0.self_attn.in_proj_weight"][start+d_head:start+2*d_head] == 1)
        # head 2 should be 0
        assert torch.all(out_sd["transformer.layers.0.self_attn.in_proj_weight"][start+2*d_head:start+3*d_head] == 0)

    # Check out_proj
    assert torch.all(out_sd["transformer.layers.0.self_attn.out_proj.weight"][:, :d_head] == 0)
    assert torch.all(out_sd["transformer.layers.0.self_attn.out_proj.weight"][:, d_head:2*d_head] == 1)
    assert torch.all(out_sd["transformer.layers.0.self_attn.out_proj.weight"][:, 2*d_head:] == 0)

    # Bias is unaffected for out_proj
    assert torch.all(out_sd["transformer.layers.0.self_attn.out_proj.bias"] == 0)

def test_patch_head_zero():
    d_model = 128
    n_heads = 4
    d_head = d_model // n_heads

    base_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(3 * d_model, d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.ones(d_model, d_model),
    }

    out_sd = patch_head(base_sd, None, 0, 2, d_model, n_heads, mode="zero")

    # Head 2 should be zeroed
    for i in range(3):
        start = i * d_model
        assert torch.all(out_sd["transformer.layers.0.self_attn.in_proj_weight"][start+2*d_head:start+3*d_head] == 0)
        assert torch.all(out_sd["transformer.layers.0.self_attn.in_proj_weight"][start:start+2*d_head] == 1)

    assert torch.all(out_sd["transformer.layers.0.self_attn.out_proj.weight"][:, 2*d_head:3*d_head] == 0)
    assert torch.all(out_sd["transformer.layers.0.self_attn.out_proj.weight"][:, :2*d_head] == 1)

def test_patch_head_random():
    d_model = 128
    n_heads = 4
    d_head = d_model // n_heads

    base_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(3 * d_model, d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.ones(d_model, d_model),
    }

    rng = torch.Generator().manual_seed(42)
    out_sd = patch_head(base_sd, None, 0, 0, d_model, n_heads, mode="random", rng=rng)

    # Original norms should be somewhat preserved due to SVD swap
    for i in range(3):
        start = i * d_model
        orig_norm = torch.norm(base_sd["transformer.layers.0.self_attn.in_proj_weight"][start:start+d_head]).item()
        new_norm = torch.norm(out_sd["transformer.layers.0.self_attn.in_proj_weight"][start:start+d_head]).item()
        assert np.isclose(orig_norm, new_norm, rtol=1e-3)

        # head 1 should still be exactly 1
        assert torch.all(out_sd["transformer.layers.0.self_attn.in_proj_weight"][start+d_head:start+2*d_head] == 1)

    orig_out_norm = torch.norm(base_sd["transformer.layers.0.self_attn.out_proj.weight"][:, :d_head]).item()
    new_out_norm = torch.norm(out_sd["transformer.layers.0.self_attn.out_proj.weight"][:, :d_head]).item()
    assert np.isclose(orig_out_norm, new_out_norm, rtol=1e-3)

def test_patch_head_invalid_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        patch_head({"transformer.layers.0.self_attn.in_proj_weight": torch.zeros(128, 128)}, None, 0, 0, 128, 4, mode="invalid")

from unittest.mock import MagicMock, patch
import numpy as np


@patch("src.transplant.head_transplant.evaluate_model")
@patch("src.transplant.head_transplant.make_loaders")
@patch("src.transplant.head_transplant.ModularArithmeticTransformer")
@patch("src.transplant.head_transplant.load_run")
def test_run_head_transplants_mocked(mock_load_run, mock_model, mock_make_loaders, mock_eval, tmp_path):
    from src.transplant.head_transplant import run_head_transplants
    from pathlib import Path

    import torch
    sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(3 * 128, 128),
        "transformer.layers.0.self_attn.in_proj_bias": torch.zeros(3 * 128),
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(128, 128),
        "transformer.layers.0.self_attn.out_proj.bias": torch.zeros(128)
    }
    mock_load_run.return_value = (sd, {"d_model": 128, "n_heads": 4, "n_layers": 1})

    mock_make_loaders.return_value = (MagicMock(), MagicMock())

    def side_effect(*args, **kwargs):
        if not hasattr(side_effect, 'count'):
            side_effect.count = 0

        c = side_effect.count
        side_effect.count += 1

        if c == 0: return {"test_acc": 0.1} # contam
        if c == 1: return {"test_acc": 0.9} # pure

        return {"test_acc": 0.5}

    mock_eval.side_effect = side_effect

    out_dir = tmp_path / "out"

    run_head_transplants([Path("pure1")], [Path("contam1")], out_dir)

    assert (out_dir / "head_transplant.json").exists()
    assert (out_dir / "head_transplant.md").exists()
    assert (out_dir / "head_rescue_heatmap.png").exists()

def test_multi_run_confidence_intervals(tmp_path):
    from src.transplant.head_transplant import run_head_transplants
    from pathlib import Path

    out_dir = tmp_path / "multi_out"

    # We can use the mock for run_head_transplants, but let's test it by mocking process_single_pair
    with patch("src.transplant.head_transplant.process_single_pair") as mock_process:
        # returns: res, greedy, mat, p_acc, c_acc
        mock_process.side_effect = [
            (
                {"L0H0": {"swap_test_acc": 0.5, "zero_test_acc": 0.1, "rand_test_acc": 0.2}},
                [((0, 0), 0.5)],
                np.array([[0.5]]),
                0.9, 0.1
            ),
            (
                {"L0H0": {"swap_test_acc": 0.7, "zero_test_acc": 0.1, "rand_test_acc": 0.2}},
                [((0, 0), 0.7)],
                np.array([[0.7]]),
                0.95, 0.15
            )
        ]

        run_head_transplants([Path("pure1"), Path("pure2")], [Path("contam1"), Path("contam2")], out_dir)

        assert (out_dir / "head_transplant.json").exists()
        import json
        with open(out_dir / "head_transplant.json") as f:
            data = json.load(f)

        assert data["mean_pure_test_acc"] == 0.925
        assert data["mean_contam_test_acc"] == 0.125
        assert len(data["runs"]) == 2
