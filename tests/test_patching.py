import torch
import pytest
from src.transplant.patching import patch_state_dict, patch_head_state_dict, keys_for

def test_keys_for_regex():
    sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(1),
        "transformer.layers.0.self_attn.in_proj_bias": torch.zeros(1),
        "transformer.layers.1.self_attn.out_proj.weight": torch.zeros(1),
        "ln.weight": torch.zeros(1)
    }

    in_proj = keys_for("self_attn_in_proj", sd)
    assert len(in_proj) == 2
    assert "transformer.layers.0.self_attn.in_proj_weight" in in_proj

    out_proj = keys_for("self_attn_out_proj", sd)
    assert len(out_proj) == 1
    assert "transformer.layers.1.self_attn.out_proj.weight" in out_proj

def test_patch_state_dict_exact():
    base_sd = {
        "ln.weight": torch.ones(5),
        "ln.bias": torch.ones(5)
    }
    donor_sd = {
        "ln.weight": torch.zeros(5),
        "ln.bias": torch.zeros(5)
    }

    patched = patch_state_dict(base_sd, donor_sd, "ln")

    assert torch.all(patched["ln.weight"] == 0)
    assert torch.all(patched["ln.bias"] == 0)

def test_patch_state_dict_shape_mismatch():
    base_sd = {"ln.weight": torch.ones(5)}
    donor_sd = {"ln.weight": torch.zeros(10)}

    with pytest.raises(ValueError, match="shape mismatch"):
        patch_state_dict(base_sd, donor_sd, "ln")

def test_patch_head_state_dict():
    d_model = 8
    n_heads = 2
    d_head = d_model // n_heads # 4

    # Fake QKV in_proj shape: (3*d_model, d_model) -> (24, 8)
    base_in = torch.ones((24, 8))
    donor_in = torch.zeros((24, 8))

    # Fake out_proj shape: (d_model, d_model) -> (8, 8)
    base_out = torch.ones((8, 8))
    donor_out = torch.zeros((8, 8))

    base_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": base_in,
        "transformer.layers.0.self_attn.out_proj.weight": base_out
    }
    donor_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": donor_in,
        "transformer.layers.0.self_attn.out_proj.weight": donor_out
    }

    # Patch Head 0
    patched_0 = patch_head_state_dict(base_sd, donor_sd, layer_idx=0, head_idx=0, d_model=d_model, n_heads=n_heads)

    # Check in_proj: Head 0 occupies rows 0-3 for Q, 8-11 for K, 16-19 for V
    in_weight_0 = patched_0["transformer.layers.0.self_attn.in_proj_weight"]

    # Q head 0
    assert torch.all(in_weight_0[0:4, :] == 0)
    # Q head 1 (untouched)
    assert torch.all(in_weight_0[4:8, :] == 1)

    # K head 0
    assert torch.all(in_weight_0[8:12, :] == 0)
    # K head 1 (untouched)
    assert torch.all(in_weight_0[12:16, :] == 1)

    # V head 0
    assert torch.all(in_weight_0[16:20, :] == 0)
    # V head 1 (untouched)
    assert torch.all(in_weight_0[20:24, :] == 1)

    # Check out_proj: Head 0 occupies cols 0-3
    out_weight_0 = patched_0["transformer.layers.0.self_attn.out_proj.weight"]
    assert torch.all(out_weight_0[:, 0:4] == 0)
    assert torch.all(out_weight_0[:, 4:8] == 1)
