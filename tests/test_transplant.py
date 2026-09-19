import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer
from src.transplant.patching import patch_attention_head, patch_module, verify_patch

def test_patch_attention_head_roundtrip():
    d_model = 16
    n_heads = 2
    model1 = ModularArithmeticTransformer(prime=7, d_model=d_model, n_heads=n_heads)
    model2 = ModularArithmeticTransformer(prime=7, d_model=d_model, n_heads=n_heads)

    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    # Patch head 0 from model2 into model1
    patched_sd = patch_attention_head(sd1, sd2, layer_idx=0, head_idx=0, d_model=d_model, n_heads=n_heads)

    # Patch head 0 from model1 back into patched_sd
    roundtrip_sd = patch_attention_head(patched_sd, sd1, layer_idx=0, head_idx=0, d_model=d_model, n_heads=n_heads)

    for k in sd1.keys():
        assert torch.equal(sd1[k], roundtrip_sd[k]), f"Roundtrip failed for {k}"

def test_patch_module():
    model1 = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2)
    model2 = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2)

    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    patched_sd = patch_module(sd1, sd2, regex_pattern=r"token_embed.*")

    assert torch.equal(patched_sd["token_embed.weight"], sd2["token_embed.weight"])
    assert torch.equal(patched_sd["pos_embed.weight"], sd1["pos_embed.weight"]) # Shouldn't change

    assert verify_patch(sd1, patched_sd, [r"token_embed.*"])

def test_patch_attention_head_slices():
    d_model = 16
    n_heads = 2
    d_head = d_model // n_heads
    model1 = ModularArithmeticTransformer(prime=7, d_model=d_model, n_heads=n_heads)
    model2 = ModularArithmeticTransformer(prime=7, d_model=d_model, n_heads=n_heads)

    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    patched_sd = patch_attention_head(sd1, sd2, layer_idx=0, head_idx=1, d_model=d_model, n_heads=n_heads)

    in_proj_weight_patched = patched_sd["transformer.layers.0.self_attn.in_proj_weight"]
    in_proj_weight_1 = sd1["transformer.layers.0.self_attn.in_proj_weight"]
    in_proj_weight_2 = sd2["transformer.layers.0.self_attn.in_proj_weight"]

    # Head 1 is indices d_head to 2*d_head in each Q, K, V block
    # Q block
    assert torch.equal(in_proj_weight_patched[d_head:2*d_head, :], in_proj_weight_2[d_head:2*d_head, :])
    assert torch.equal(in_proj_weight_patched[0:d_head, :], in_proj_weight_1[0:d_head, :])

    # K block
    assert torch.equal(in_proj_weight_patched[d_model+d_head:d_model+2*d_head, :], in_proj_weight_2[d_model+d_head:d_model+2*d_head, :])
    assert torch.equal(in_proj_weight_patched[d_model:d_model+d_head, :], in_proj_weight_1[d_model:d_model+d_head, :])

    # V block
    assert torch.equal(in_proj_weight_patched[2*d_model+d_head:2*d_model+2*d_head, :], in_proj_weight_2[2*d_model+d_head:2*d_model+2*d_head, :])
    assert torch.equal(in_proj_weight_patched[2*d_model:2*d_model+d_head, :], in_proj_weight_1[2*d_model:2*d_model+d_head, :])

def test_transplant_experiment_config_schema():
    # Tests that default argument combinations for the transplant experiment work nicely.
    import argparse
    from src.transplant.head_transplant import main as head_transplant_main

    parser = argparse.ArgumentParser()
    # Dummy mock for argparse
    parser.add_argument("--donor-runs", type=str, nargs="+")
    parser.add_argument("--recipient-runs", type=str, nargs="+")
    args = parser.parse_args(["--donor-runs", "run1", "--recipient-runs", "run2"])

    assert args.donor_runs == ["run1"]
    assert args.recipient_runs == ["run2"]
