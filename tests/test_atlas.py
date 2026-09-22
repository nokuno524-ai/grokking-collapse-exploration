import pytest
import torch
from src.transplant.atlas import (
    apply_intervention_to_tensor,
    patch_attention_head,
    patch_mlp_neuron,
    patch_layer,
    get_head_indices
)

def test_get_head_indices():
    start, end = get_head_indices(head_idx=1, d_model=128, n_heads=4)
    # d_head = 32. Head 1 should be 32 to 64
    assert start == 32
    assert end == 64

def test_apply_intervention_to_tensor_zero():
    tensor = torch.ones(10, 5)
    # Zero out rows 2 to 4 (dim=0)
    modified = apply_intervention_to_tensor(tensor, None, "zero", dim=0, indices=(2, 4))

    assert torch.all(modified[0:2] == 1.0)
    assert torch.all(modified[2:4] == 0.0)
    assert torch.all(modified[4:] == 1.0)

def test_apply_intervention_to_tensor_mean():
    tensor = torch.arange(10, dtype=torch.float).unsqueeze(1).expand(-1, 5).clone()
    # tensor has rows with values 0, 1, 2, ..., 9

    # Mean ablation on rows 2 to 4.
    # The mean of values (2, 3) across 5 columns is 2.5
    modified = apply_intervention_to_tensor(tensor, None, "mean", dim=0, indices=(2, 4))

    assert modified[2, 0].item() == 2.5
    assert modified[3, 0].item() == 2.5
    assert modified[4, 0].item() == 4.0 # Unchanged

def test_apply_intervention_to_tensor_transplant():
    tensor = torch.ones(10, 5)
    donor = torch.zeros(10, 5)

    # Transplant rows 2 to 4 (dim=0) from donor to tensor
    modified = apply_intervention_to_tensor(tensor, donor, "transplant", dim=0, indices=(2, 4))

    assert torch.all(modified[0:2] == 1.0)
    assert torch.all(modified[2:4] == 0.0)
    assert torch.all(modified[4:] == 1.0)

def test_patch_attention_head():
    d_model = 128
    n_heads = 4
    d_head = 32

    base_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(3 * d_model, d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.ones(d_model, d_model)
    }

    # Zero out Head 1 (indices 32 to 64 in each Q, K, V block)
    patched = patch_attention_head(
        base_sd, None, layer_idx=0, head_idx=1, d_model=d_model, n_heads=n_heads, intervention="zero"
    )

    in_proj = patched["transformer.layers.0.self_attn.in_proj_weight"]
    out_proj = patched["transformer.layers.0.self_attn.out_proj.weight"]

    # Check in_proj
    # Q part: Head 1 is 32 to 64
    assert torch.all(in_proj[32:64] == 0.0)
    assert torch.all(in_proj[0:32] == 1.0)
    # K part: Head 1 is d_model + 32 to d_model + 64 (160 to 192)
    assert torch.all(in_proj[160:192] == 0.0)
    assert torch.all(in_proj[128:160] == 1.0)
    # V part: Head 1 is 2*d_model + 32 to 2*d_model + 64 (288 to 320)
    assert torch.all(in_proj[288:320] == 0.0)
    assert torch.all(in_proj[256:288] == 1.0)

    # Check out_proj (dim=1)
    assert torch.all(out_proj[:, 32:64] == 0.0)
    assert torch.all(out_proj[:, 0:32] == 1.0)

def test_patch_mlp_neuron():
    base_sd = {
        "transformer.layers.0.linear1.weight": torch.ones(512, 128),
        "transformer.layers.0.linear2.weight": torch.ones(128, 512)
    }

    # Zero out neuron 5
    patched = patch_mlp_neuron(base_sd, None, layer_idx=0, neuron_idx=5, intervention="zero")

    w1 = patched["transformer.layers.0.linear1.weight"]
    w2 = patched["transformer.layers.0.linear2.weight"]

    # linear1: neuron is dim 0
    assert torch.all(w1[5] == 0.0)
    assert torch.all(w1[4] == 1.0)
    assert torch.all(w1[6] == 1.0)

    # linear2: neuron is dim 1
    assert torch.all(w2[:, 5] == 0.0)
    assert torch.all(w2[:, 4] == 1.0)
    assert torch.all(w2[:, 6] == 1.0)

def test_patch_layer():
    base_sd = {
        "transformer.layers.0.linear1.weight": torch.ones(10, 10),
        "transformer.layers.1.linear1.weight": torch.ones(10, 10) # Different layer
    }

    patched = patch_layer(base_sd, None, layer_idx=0, intervention="zero")

    assert torch.all(patched["transformer.layers.0.linear1.weight"] == 0.0)
    assert torch.all(patched["transformer.layers.1.linear1.weight"] == 1.0)
