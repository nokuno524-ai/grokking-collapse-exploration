import torch
import pytest
from src.transplant.patching import keys_for, patch_state_dict, patch_head_state_dict
from src.transplant.transplant_rescue import COMPONENT_PATTERNS
from src.transplant.sweep import discover_runs, pair_runs

def test_keys_for():
    sd = {
        "token_embed.weight": torch.randn(10, 10),
        "transformer.layers.0.self_attn.in_proj_weight": torch.randn(30, 10),
        "transformer.layers.0.linear1.weight": torch.randn(40, 10),
        "transformer.layers.1.linear1.weight": torch.randn(40, 10),
        "other": torch.randn(10)
    }

    assert keys_for("token_embed", sd, COMPONENT_PATTERNS) == ["token_embed.weight"]
    assert "transformer.layers.0.self_attn.in_proj_weight" in keys_for("self_attn_in_proj", sd, COMPONENT_PATTERNS)
    assert keys_for("linear1", sd, COMPONENT_PATTERNS) == ["transformer.layers.0.linear1.weight", "transformer.layers.1.linear1.weight"]

def test_patch_state_dict_weight_swapping():
    base = {
        "token_embed.weight": torch.zeros(10, 10),
        "other": torch.zeros(10)
    }
    donor = {
        "token_embed.weight": torch.ones(10, 10),
        "other": torch.ones(10)
    }

    out = patch_state_dict(base, donor, "token_embed", COMPONENT_PATTERNS)
    # Weights swapped
    assert torch.all(out["token_embed.weight"] == 1.0)
    # Others preserved
    assert torch.all(out["other"] == 0.0)

def test_patch_state_dict_shape_mismatches():
    base = {
        "token_embed.weight": torch.zeros(10, 10)
    }
    donor = {
        "token_embed.weight": torch.ones(10, 20)
    }

    with pytest.raises(ValueError, match="shape mismatch"):
        patch_state_dict(base, donor, "token_embed", COMPONENT_PATTERNS)

def test_patch_head_state_dict_qkv_slicing():
    d_model = 12
    n_heads = 3
    d_head = 4

    # base is 0, donor is 1
    base = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(3 * d_model, d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(d_model, d_model),
        "other": torch.zeros(10)
    }

    donor = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(3 * d_model, d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.ones(d_model, d_model),
        "other": torch.ones(10)
    }

    out = patch_head_state_dict(base, donor, layer_idx=0, head_idx=1, d_model=d_model, n_heads=n_heads)

    # Check in_proj
    in_w = out["transformer.layers.0.self_attn.in_proj_weight"]
    for i in range(3): # Q, K, V
        for h in range(n_heads):
            start = i * d_model + h * d_head
            end = start + d_head
            if h == 1:
                assert torch.all(in_w[start:end] == 1.0) # Patched head
            else:
                assert torch.all(in_w[start:end] == 0.0) # Base heads

    # Check out_proj
    out_w = out["transformer.layers.0.self_attn.out_proj.weight"]
    for h in range(n_heads):
        start = h * d_head
        end = start + d_head
        if h == 1:
            assert torch.all(out_w[:, start:end] == 1.0) # Patched head
        else:
            assert torch.all(out_w[:, start:end] == 0.0) # Base heads

def test_matrix_assembly_determinism(tmp_path):
    # Setup dummy run structure
    pure_dir = tmp_path / "pure"
    pure_dir.mkdir()
    (pure_dir / "seed_42").mkdir()
    (pure_dir / "seed_42" / "checkpoint_000.pt").touch()

    contam_dir = tmp_path / "contam"
    contam_dir.mkdir()
    (contam_dir / "seed_42").mkdir()
    (contam_dir / "seed_42" / "checkpoint_000.pt").touch()

    runs = discover_runs(tmp_path)
    assert len(runs) == 2

    pairs = pair_runs(runs)
    # 2 conditions = 4 possible pairs (pure->pure, pure->contam, contam->pure, contam->contam)
    assert len(pairs) == 4

    # Ensure they are consistently ordered
    assert pairs[0][0] == "contam" and pairs[0][1] == "contam"
    assert pairs[1][0] == "contam" and pairs[1][1] == "pure"
    assert pairs[2][0] == "pure" and pairs[2][1] == "contam"
    assert pairs[3][0] == "pure" and pairs[3][1] == "pure"
