import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer
from src.transplant.patching import patch_state_dict, patch_head_state_dict
from src.transplant_rescue import COMPONENT_PATTERNS

@pytest.fixture
def base_model():
    torch.manual_seed(42)
    return ModularArithmeticTransformer(prime=59, d_model=32, n_heads=4, n_layers=2)

@pytest.fixture
def donor_model():
    torch.manual_seed(43)
    return ModularArithmeticTransformer(prime=59, d_model=32, n_heads=4, n_layers=2)

def test_patch_state_dict(base_model, donor_model):
    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    # Check that initially they are different
    assert not torch.allclose(base_sd['token_embed.weight'], donor_sd['token_embed.weight'])

    # Patch token embed
    patched_sd = patch_state_dict(base_sd, donor_sd, 'token_embed')

    # Verify patch
    assert torch.allclose(patched_sd['token_embed.weight'], donor_sd['token_embed.weight'])

    # Verify rest is unchanged
    assert torch.allclose(patched_sd['pos_embed.weight'], base_sd['pos_embed.weight'])

def test_patch_head_state_dict(base_model, donor_model):
    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    d_model = 32
    n_heads = 4
    d_head = d_model // n_heads
    layer_idx = 0
    head_idx = 1

    patched_sd = patch_head_state_dict(base_sd, donor_sd, layer_idx, head_idx, d_model, n_heads)

    prefix = f"transformer.layers.{layer_idx}.self_attn."
    in_proj_w = f"{prefix}in_proj_weight"

    start = head_idx * d_head
    end = (head_idx + 1) * d_head

    # Q part of in_proj_weight
    assert torch.allclose(
        patched_sd[in_proj_w][0 + start:0 + end, :],
        donor_sd[in_proj_w][0 + start:0 + end, :]
    )

    # K part
    assert torch.allclose(
        patched_sd[in_proj_w][d_model + start:d_model + end, :],
        donor_sd[in_proj_w][d_model + start:d_model + end, :]
    )

    # V part
    assert torch.allclose(
        patched_sd[in_proj_w][2 * d_model + start:2 * d_model + end, :],
        donor_sd[in_proj_w][2 * d_model + start:2 * d_model + end, :]
    )

    # Unpatched head 0 should remain base
    assert torch.allclose(
        patched_sd[in_proj_w][0:d_head, :],
        base_sd[in_proj_w][0:d_head, :]
    )

def test_patch_state_dict_errors(base_model, donor_model):
    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    # Unknown component
    with pytest.raises(ValueError, match="unknown component"):
        patch_state_dict(base_sd, donor_sd, "nonexistent_component")

    # Shape mismatch
    # Create a mismatched donor
    bad_donor_sd = {k: v.clone() for k, v in donor_sd.items()}
    bad_donor_sd['token_embed.weight'] = torch.randn(10, 10)
    with pytest.raises(ValueError, match="shape mismatch"):
        patch_state_dict(base_sd, bad_donor_sd, "token_embed")

def test_patch_head_state_dict_errors(base_model, donor_model):
    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    # Indivisible d_model
    with pytest.raises(ValueError, match="must be divisible"):
        patch_head_state_dict(base_sd, donor_sd, layer_idx=0, head_idx=0, d_model=31, n_heads=4)
