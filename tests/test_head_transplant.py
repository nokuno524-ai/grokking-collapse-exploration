import torch
from typing import Tuple
from src.transplant.head_transplant import (
    slice_qkv_weight,
    inject_qkv_weight,
    inject_out_proj_weight,
    transplant_head
)

def test_slice_qkv_weight():
    d_model = 8
    n_heads = 2
    d_head = 4

    # Create an in_proj weight matrix
    in_proj = torch.arange(3 * d_model * d_model).float().reshape(3 * d_model, d_model)

    q_w, k_w, v_w = slice_qkv_weight(in_proj, d_model, n_heads, 1)

    assert q_w.shape == (d_head, d_model)
    assert k_w.shape == (d_head, d_model)
    assert v_w.shape == (d_head, d_model)

    # Check values. Q starts at 0. Head 1 starts at row 4
    assert torch.allclose(q_w[0], in_proj[4])
    # K starts at 8. Head 1 starts at 12
    assert torch.allclose(k_w[0], in_proj[12])
    # V starts at 16. Head 1 starts at 20
    assert torch.allclose(v_w[0], in_proj[20])

def test_inject_qkv_weight():
    d_model = 8
    n_heads = 2

    base = torch.zeros(3 * d_model, d_model)
    donor = torch.ones(3 * d_model, d_model)

    new_weight = inject_qkv_weight(base, donor, d_model, n_heads, 0)

    # Check that Q head 0 is all ones
    assert torch.all(new_weight[:4, :] == 1)
    # Check that Q head 1 is all zeros
    assert torch.all(new_weight[4:8, :] == 0)
    # Check K head 0 is all ones
    assert torch.all(new_weight[8:12, :] == 1)

def test_inject_out_proj_weight():
    d_model = 8
    n_heads = 2

    base = torch.zeros(d_model, d_model)
    donor = torch.ones(d_model, d_model)

    new_weight = inject_out_proj_weight(base, donor, d_model, n_heads, 1)

    # Columns 0-3 should be 0, columns 4-7 should be 1
    assert torch.all(new_weight[:, :4] == 0)
    assert torch.all(new_weight[:, 4:] == 1)

def test_transplant_head():
    d_model = 8
    n_heads = 2
    layer_idx = 0
    head_idx = 0

    base_sd = {
        f"transformer.layers.0.self_attn.in_proj_weight": torch.zeros(3 * d_model, d_model),
        f"transformer.layers.0.self_attn.in_proj_bias": torch.zeros(3 * d_model),
        f"transformer.layers.0.self_attn.out_proj.weight": torch.zeros(d_model, d_model)
    }

    donor_sd = {
        f"transformer.layers.0.self_attn.in_proj_weight": torch.ones(3 * d_model, d_model),
        f"transformer.layers.0.self_attn.in_proj_bias": torch.ones(3 * d_model),
        f"transformer.layers.0.self_attn.out_proj.weight": torch.ones(d_model, d_model)
    }

    new_sd = transplant_head(base_sd, donor_sd, layer_idx, head_idx, d_model, n_heads)

    assert torch.all(new_sd[f"transformer.layers.0.self_attn.in_proj_weight"][:4, :] == 1)
    assert torch.all(new_sd[f"transformer.layers.0.self_attn.in_proj_weight"][4:8, :] == 0)
    assert torch.all(new_sd[f"transformer.layers.0.self_attn.out_proj.weight"][:, :4] == 1)
    assert torch.all(new_sd[f"transformer.layers.0.self_attn.out_proj.weight"][:, 4:] == 0)
