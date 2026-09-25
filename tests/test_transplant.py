import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer
from src.transplant.patching import patch_state_dict, patch_head_state_dict
from src.transplant.sweep import graft_context

def test_patch_state_dict_dtype_device():
    base_model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)
    donor_model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)

    # Intentionally change donor dtype
    donor_sd = {k: v.to(torch.float64) for k, v in donor_model.state_dict().items()}
    base_sd = base_model.state_dict()

    patched_sd = patch_state_dict(base_sd, donor_sd, "token_embed")

    assert patched_sd["token_embed.weight"].dtype == base_sd["token_embed.weight"].dtype
    assert not torch.allclose(patched_sd["token_embed.weight"], base_sd["token_embed.weight"])

def test_patch_head_state_dict():
    base_model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)
    donor_model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)

    # Fill donor with 1s, base with 0s for easy checking
    donor_sd = {k: torch.ones_like(v) for k, v in donor_model.state_dict().items()}
    base_sd = {k: torch.zeros_like(v) for k, v in base_model.state_dict().items()}

    patched_sd = patch_head_state_dict(base_sd, donor_sd, layer_idx=0, head_idx=1, d_head=8, num_heads=2)

    # For out_proj_weight, only the specific head's columns should be patched
    out_proj_w = patched_sd["transformer.layers.0.self_attn.out_proj.weight"]
    assert torch.all(out_proj_w[:, 8:16] == 1.0)
    assert torch.all(out_proj_w[:, 0:8] == 0.0)

def test_graft_context_round_trip():
    model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)
    donor_model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)

    orig_sd = {k: v.clone() for k, v in model.state_dict().items()}

    with graft_context(model, donor_model.state_dict(), ["token_embed"]):
        assert not torch.allclose(model.state_dict()["token_embed.weight"], orig_sd["token_embed.weight"])

    # Check if restored
    for k, v in model.state_dict().items():
        assert torch.allclose(v, orig_sd[k])

def test_patch_state_dict_key_mapping():
    base_model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)
    donor_model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)

    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    # Intentionally rename a key in donor to simulate a mapping need
    donor_sd["old_token_embed"] = donor_sd.pop("token_embed.weight")

    # Without mapping, it should fail
    with pytest.raises(KeyError):
        patch_state_dict(base_sd, donor_sd, "token_embed")

    # With mapping, it should succeed
    mapping = {"old_token_embed": "token_embed.weight"}
    patched = patch_state_dict(base_sd, donor_sd, "token_embed", key_mapping=mapping)

    assert torch.allclose(patched["token_embed.weight"], donor_sd["old_token_embed"])

def test_patch_state_dict_shape_mismatch():
    base_model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)
    donor_model = ModularArithmeticTransformer(d_model=32, n_heads=2, d_ff=32, n_layers=2) # Different shape

    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    with pytest.raises(ValueError, match="shape mismatch"):
        patch_state_dict(base_sd, donor_sd, "token_embed")
