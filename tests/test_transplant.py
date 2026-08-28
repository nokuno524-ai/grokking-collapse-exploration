import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer
from src.transplant.circuits import swap_weights, swap_attention_head, swap_mlp, swap_layer_norm

@pytest.fixture
def base_model():
    return ModularArithmeticTransformer(
        prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=1
    )

@pytest.fixture
def donor_model():
    m = ModularArithmeticTransformer(
        prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=1
    )
    # Ensure they have different weights
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    return m

def test_transplant_reversibility(base_model, donor_model):
    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    # Swap head 0
    swapped_sd = swap_attention_head(base_sd, donor_sd, 0, 0, 16, 2)
    # Swap it back from base
    swapped_back_sd = swap_attention_head(swapped_sd, base_sd, 0, 0, 16, 2)

    # Check that swapped_back matches original base
    for k in base_sd:
        assert torch.allclose(base_sd[k], swapped_back_sd[k]), f"Mismatch in {k} after reverse swap"

def test_no_leakage(base_model, donor_model):
    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    # Swap layer norm
    swapped_sd = swap_layer_norm(base_sd, donor_sd, 0)

    ln_keys = {
        "transformer.layers.0.norm1.weight",
        "transformer.layers.0.norm1.bias",
        "transformer.layers.0.norm2.weight",
        "transformer.layers.0.norm2.bias"
    }

    for k in base_sd:
        if k in ln_keys:
            assert torch.allclose(swapped_sd[k], donor_sd[k])
        else:
            assert torch.allclose(swapped_sd[k], base_sd[k])

def test_shape_mismatch(base_model):
    base_sd = base_model.state_dict()

    # Create donor with different d_model
    donor_model = ModularArithmeticTransformer(
        prime=5, d_model=32, n_heads=2, d_ff=32, n_layers=1
    )
    donor_sd = donor_model.state_dict()

    with pytest.raises(RuntimeError, match="Shape mismatch"):
        swap_weights(base_sd, donor_sd, {"transformer.layers.0.self_attn.in_proj_weight"})

    with pytest.raises(RuntimeError):
        swap_attention_head(base_sd, donor_sd, 0, 0, 16, 2)

def test_eval_run(base_model, donor_model):
    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    swapped_sd = swap_mlp(base_sd, donor_sd, 0)

    # This acts as our strict check test
    model = ModularArithmeticTransformer(
        prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=1
    )

    # Must use strict=True per user instructions
    model.load_state_dict(swapped_sd, strict=True)

    # Forward pass works
    x = torch.tensor([[1, 2], [3, 4]])
    out = model(x)
    assert out.shape == (2, 5)
