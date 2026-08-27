import pytest
import torch
import copy
from src.model import ModularArithmeticTransformer
from src.transplant_rescue import patch_state_dict, keys_for

@pytest.fixture
def base_model():
    return ModularArithmeticTransformer(prime=59, d_model=16, n_heads=2, d_ff=32, n_layers=2)

@pytest.fixture
def donor_model():
    return ModularArithmeticTransformer(prime=59, d_model=16, n_heads=2, d_ff=32, n_layers=2)

def test_transplant_head_components(base_model, donor_model):
    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    # Check that keys_for resolves layer_1_head_0 properly
    keys = keys_for("layer_1_head_0", base_sd)
    assert any("in_proj_weight" in k for k in keys)
    assert any("out_proj.weight" in k for k in keys)

    patched_sd = patch_state_dict(base_sd, donor_sd, "layer_1_head_0", n_heads=2)

    # Check that original base_sd is NOT mutated
    assert torch.all(base_sd["transformer.layers.1.self_attn.in_proj_weight"] == base_model.state_dict()["transformer.layers.1.self_attn.in_proj_weight"])

    d_model = 16
    head_dim = 8

    in_proj_base = base_sd["transformer.layers.1.self_attn.in_proj_weight"]
    in_proj_donor = donor_sd["transformer.layers.1.self_attn.in_proj_weight"]
    in_proj_patched = patched_sd["transformer.layers.1.self_attn.in_proj_weight"]

    # For head 0, Q, K, V are at indices [0:8], [16:24], [32:40]
    # Check Q
    assert torch.all(in_proj_patched[0:8, :] == in_proj_donor[0:8, :])
    assert torch.all(in_proj_patched[8:16, :] == in_proj_base[8:16, :])

    # Check K
    assert torch.all(in_proj_patched[16:24, :] == in_proj_donor[16:24, :])
    assert torch.all(in_proj_patched[24:32, :] == in_proj_base[24:32, :])

    # Check V
    assert torch.all(in_proj_patched[32:40, :] == in_proj_donor[32:40, :])
    assert torch.all(in_proj_patched[40:48, :] == in_proj_base[40:48, :])


    # Check out_proj
    out_proj_base = base_sd["transformer.layers.1.self_attn.out_proj.weight"]
    out_proj_donor = donor_sd["transformer.layers.1.self_attn.out_proj.weight"]
    out_proj_patched = patched_sd["transformer.layers.1.self_attn.out_proj.weight"]

    # For out_proj, head 0 inputs are at columns [0:8]
    assert torch.all(out_proj_patched[:, 0:8] == out_proj_donor[:, 0:8])
    assert torch.all(out_proj_patched[:, 8:16] == out_proj_base[:, 8:16])

    # Check in_proj_bias
    in_proj_bias_base = base_sd["transformer.layers.1.self_attn.in_proj_bias"]
    in_proj_bias_donor = donor_sd["transformer.layers.1.self_attn.in_proj_bias"]
    in_proj_bias_patched = patched_sd["transformer.layers.1.self_attn.in_proj_bias"]

    # Q bias
    assert torch.all(in_proj_bias_patched[0:8] == in_proj_bias_donor[0:8])
    assert torch.all(in_proj_bias_patched[8:16] == in_proj_bias_base[8:16])
    # K bias
    assert torch.all(in_proj_bias_patched[16:24] == in_proj_bias_donor[16:24])
    assert torch.all(in_proj_bias_patched[24:32] == in_proj_bias_base[24:32])
    # V bias
    assert torch.all(in_proj_bias_patched[32:40] == in_proj_bias_donor[32:40])
    assert torch.all(in_proj_bias_patched[40:48] == in_proj_bias_base[40:48])


def test_randomize_head(base_model):
    base_sd = base_model.state_dict()

    patched_sd = patch_state_dict(base_sd, {}, "layer_0_head_1", randomize=True, n_heads=2)

    in_proj_base = base_sd["transformer.layers.0.self_attn.in_proj_weight"]
    in_proj_patched = patched_sd["transformer.layers.0.self_attn.in_proj_weight"]

    # Head 1 Q slice: [8:16]
    # Should not match base
    assert not torch.allclose(in_proj_patched[8:16, :], in_proj_base[8:16, :])
    # Head 0 Q slice: [0:8]
    # Should match base exactly
    assert torch.all(in_proj_patched[0:8, :] == in_proj_base[0:8, :])
