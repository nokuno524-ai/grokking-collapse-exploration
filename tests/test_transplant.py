import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer
from src.transplant.head_transplant import patch_head, split_qkv, splice_head_weight, patch_mlp, patch_layer

def test_split_qkv():
    d_model = 128
    weight = torch.randn(3 * d_model, d_model)
    q, k, v = split_qkv(weight, d_model)
    assert q.shape == (d_model, d_model)
    assert k.shape == (d_model, d_model)
    assert v.shape == (d_model, d_model)
    assert torch.allclose(q, weight[:d_model])
    assert torch.allclose(k, weight[d_model:2*d_model])
    assert torch.allclose(v, weight[2*d_model:])

def test_patch_head():
    cfg = {"d_model": 128, "n_heads": 4, "n_layers": 1}
    model_base = ModularArithmeticTransformer(**cfg)
    model_donor = ModularArithmeticTransformer(**cfg)

    # ensure different weights
    for p in model_donor.parameters():
        p.data.add_(1.0)

    base_sd = model_base.state_dict()
    donor_sd = model_donor.state_dict()

    layer_idx = 0
    head_idx = 1 # head 1 (index 1 of 4)
    d_model = 128
    n_heads = 4

    patched_sd = patch_head(base_sd, donor_sd, layer_idx, head_idx, n_heads, d_model)

    prefix = f"transformer.layers.{layer_idx}.self_attn"

    # Check in_proj_weight
    in_w_patched = patched_sd[f"{prefix}.in_proj_weight"]
    in_w_base = base_sd[f"{prefix}.in_proj_weight"]
    in_w_donor = donor_sd[f"{prefix}.in_proj_weight"]

    q_p, k_p, v_p = split_qkv(in_w_patched, d_model)
    q_b, k_b, v_b = split_qkv(in_w_base, d_model)
    q_d, k_d, v_d = split_qkv(in_w_donor, d_model)

    head_dim = d_model // n_heads
    start = head_idx * head_dim
    end = start + head_dim

    # Check donor part
    assert torch.allclose(q_p[start:end], q_d[start:end])
    assert torch.allclose(k_p[start:end], k_d[start:end])
    assert torch.allclose(v_p[start:end], v_d[start:end])

    # Check base part (e.g. head 0)
    assert torch.allclose(q_p[0:start], q_b[0:start])
    assert torch.allclose(q_p[end:], q_b[end:])

    # out_proj weight
    out_w_patched = patched_sd[f"{prefix}.out_proj.weight"]
    out_w_base = base_sd[f"{prefix}.out_proj.weight"]
    out_w_donor = donor_sd[f"{prefix}.out_proj.weight"]

    assert torch.allclose(out_w_patched[:, start:end], out_w_donor[:, start:end])
    assert torch.allclose(out_w_patched[:, 0:start], out_w_base[:, 0:start])

    # Check that unrelated weights are unmodified
    assert torch.allclose(patched_sd["token_embed.weight"], base_sd["token_embed.weight"])

def test_patch_mlp():
    cfg = {"d_model": 128, "n_heads": 4, "n_layers": 1}
    model_base = ModularArithmeticTransformer(**cfg)
    model_donor = ModularArithmeticTransformer(**cfg)

    # ensure different weights
    for p in model_donor.parameters():
        p.data.add_(1.0)

    base_sd = model_base.state_dict()
    donor_sd = model_donor.state_dict()

    patched_sd = patch_mlp(base_sd, donor_sd, 0)

    assert torch.allclose(patched_sd["transformer.layers.0.linear1.weight"], donor_sd["transformer.layers.0.linear1.weight"])
    assert torch.allclose(patched_sd["transformer.layers.0.linear2.weight"], donor_sd["transformer.layers.0.linear2.weight"])
    # Not part of MLP
    assert torch.allclose(patched_sd["transformer.layers.0.norm1.weight"], base_sd["transformer.layers.0.norm1.weight"])

def test_patch_layer():
    cfg = {"d_model": 128, "n_heads": 4, "n_layers": 1}
    model_base = ModularArithmeticTransformer(**cfg)
    model_donor = ModularArithmeticTransformer(**cfg)

    for p in model_donor.parameters():
        p.data.add_(1.0)

    base_sd = model_base.state_dict()
    donor_sd = model_donor.state_dict()

    patched_sd = patch_layer(base_sd, donor_sd, 0)

    assert torch.allclose(patched_sd["transformer.layers.0.linear1.weight"], donor_sd["transformer.layers.0.linear1.weight"])
    assert torch.allclose(patched_sd["transformer.layers.0.self_attn.in_proj_weight"], donor_sd["transformer.layers.0.self_attn.in_proj_weight"])
    assert torch.allclose(patched_sd["token_embed.weight"], base_sd["token_embed.weight"])
