import pytest
import torch
import torch.nn as nn
from src.transplant.transplant_rescue import patch_state_dict

def test_patch_state_dict():
    # Setup
    base_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(10, 10),
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(10, 10),
        "transformer.layers.0.linear1.weight": torch.zeros(10, 10),
    }
    donor_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(10, 10),
        "transformer.layers.0.self_attn.out_proj.weight": torch.ones(10, 10),
        "transformer.layers.0.linear1.weight": torch.ones(10, 10),
    }

    # Test patch linear1
    patched_sd = patch_state_dict(base_sd, donor_sd, "linear1")
    assert torch.all(patched_sd["transformer.layers.0.linear1.weight"] == 1.0)
    assert torch.all(patched_sd["transformer.layers.0.self_attn.in_proj_weight"] == 0.0)

from src.transplant.patching import patch_head_state_dict, patch_layer_component_state_dict

def test_patch_head_state_dict():
    n_heads = 4
    d_head = 8
    d_model = n_heads * d_head

    # Initialize some dummy weights
    # QKV are concatenated in in_proj_weight, so shape is (3 * d_model, d_model)
    base_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(3 * d_model, d_model),
        "transformer.layers.0.self_attn.in_proj_bias": torch.zeros(3 * d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(d_model, d_model),
    }
    donor_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(3 * d_model, d_model),
        "transformer.layers.0.self_attn.in_proj_bias": torch.ones(3 * d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.ones(d_model, d_model),
    }

    # Patch head 1 (indices 8 to 15 out of 32 for Q, K, V)
    patched = patch_head_state_dict(base_sd, donor_sd, layer_idx=0, head_idx=1, d_head=d_head, n_heads=n_heads)

    # Check Q slice (head 1 is idx 8..15)
    assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][8:16] == 1.0)
    assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][0:8] == 0.0)
    assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][16:32] == 0.0)

    # Check K slice (head 1 is offset by d_model = 32 -> 40..47)
    assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][40:48] == 1.0)
    assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][32:40] == 0.0)

    # Check V slice (head 1 is offset by 2*d_model = 64 -> 72..79)
    assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][72:80] == 1.0)
    assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][64:72] == 0.0)

    # Check out_proj (head 1 corresponds to input columns 8..15)
    assert torch.all(patched["transformer.layers.0.self_attn.out_proj.weight"][:, 8:16] == 1.0)
    assert torch.all(patched["transformer.layers.0.self_attn.out_proj.weight"][:, 0:8] == 0.0)
    assert torch.all(patched["transformer.layers.0.self_attn.out_proj.weight"][:, 16:] == 0.0)


def test_patch_layer_component_state_dict():
    base_sd = {
        "transformer.layers.0.linear1.weight": torch.zeros(10, 10),
        "transformer.layers.0.linear2.weight": torch.zeros(10, 10),
        "transformer.layers.1.linear1.weight": torch.zeros(10, 10),
    }
    donor_sd = {
        "transformer.layers.0.linear1.weight": torch.ones(10, 10),
        "transformer.layers.0.linear2.weight": torch.ones(10, 10),
        "transformer.layers.1.linear1.weight": torch.ones(10, 10),
    }

    # Patch only Layer 0's MLP (linear1 and linear2)
    patched = patch_layer_component_state_dict(base_sd, donor_sd, layer_idx=0, component_name="mlp")

    assert torch.all(patched["transformer.layers.0.linear1.weight"] == 1.0)
    assert torch.all(patched["transformer.layers.0.linear2.weight"] == 1.0)
    # Layer 1 should remain base (zeros)
    assert torch.all(patched["transformer.layers.1.linear1.weight"] == 0.0)

def test_shape_mismatch():
    base_sd = {"transformer.layers.0.linear1.weight": torch.zeros(10, 10)}
    donor_sd = {"transformer.layers.0.linear1.weight": torch.ones(5, 5)}  # Mismatched shape

    with pytest.raises(ValueError, match="shape mismatch"):
        patch_layer_component_state_dict(base_sd, donor_sd, layer_idx=0, component_name="mlp")


def test_missing_key_in_donor():
    base_sd = {
        "transformer.layers.0.linear1.weight": torch.zeros(10, 10),
        "transformer.layers.0.linear2.weight": torch.zeros(10, 10),
    }
    # Donor is missing linear2
    donor_sd = {
        "transformer.layers.0.linear1.weight": torch.ones(10, 10),
    }

    patched = patch_layer_component_state_dict(base_sd, donor_sd, layer_idx=0, component_name="mlp")

    assert torch.all(patched["transformer.layers.0.linear1.weight"] == 1.0)
    # linear2 should be untouched since donor is missing it
    assert torch.all(patched["transformer.layers.0.linear2.weight"] == 0.0)

from src.model import ModularArithmeticTransformer
from src.transplant.sweep import ComponentResult

def test_toy_transformer_shapes():
    model = ModularArithmeticTransformer(
        prime=5,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=2
    )
    sd = model.state_dict()
    donor = {k: v.clone() * 2 for k, v in sd.items()}

    # Test head patch on real model dict
    patched_h = patch_head_state_dict(sd, donor, 0, 0, 8, 2)
    # Check that in_proj_weight has a specific slice doubled
    assert torch.all(patched_h["transformer.layers.0.self_attn.in_proj_weight"][0:8] == sd["transformer.layers.0.self_attn.in_proj_weight"][0:8] * 2)

    # Test mlp patch on real model dict
    patched_mlp = patch_layer_component_state_dict(sd, donor, 1, "mlp")
    assert torch.all(patched_mlp["transformer.layers.1.linear1.weight"] == sd["transformer.layers.1.linear1.weight"] * 2)
    assert torch.all(patched_mlp["transformer.layers.1.linear2.weight"] == sd["transformer.layers.1.linear2.weight"] * 2)
    assert torch.all(patched_mlp["transformer.layers.0.linear1.weight"] == sd["transformer.layers.0.linear1.weight"])

from src.transplant.transplant_rescue import load_run
import tempfile
import json
from pathlib import Path

def test_missing_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        with pytest.raises(FileNotFoundError):
            load_run(p)
