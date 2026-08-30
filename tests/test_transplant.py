import pytest
import torch
import torch.nn as nn
from src.transplant.circuits import patch_state_dict, patch_state_dict_fractional, clean_key

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(10, 16)
        self.pos_embed = nn.Embedding(10, 16)
        self.transformer = nn.ModuleDict({
            "layers": nn.ModuleList([
                nn.ModuleDict({
                    "self_attn": nn.Linear(16, 16),
                    "linear1": nn.Linear(16, 32),
                    "linear2": nn.Linear(32, 16)
                })
            ])
        })
        self.output_head = nn.Linear(16, 10)

def test_clean_key():
    assert clean_key("module.weight") == "weight"
    assert clean_key("_orig_mod.weight") == "weight"
    assert clean_key("module._orig_mod.weight") == "weight"
    assert clean_key("weight") == "weight"

def test_patch_state_dict_shape_mismatch():
    base_model = DummyModel()
    donor_model = DummyModel()

    # Introduce shape mismatch
    donor_model.token_embed = nn.Embedding(12, 16)

    with pytest.raises(ValueError, match="shape mismatch"):
        patch_state_dict(base_model.state_dict(), donor_model.state_dict(), "token_embed")

def test_patch_state_dict_reversible():
    base_model = DummyModel()
    donor_model = DummyModel()

    # Change weights so they are different
    with torch.no_grad():
        donor_model.token_embed.weight.add_(1.0)

    base_sd = base_model.state_dict()
    donor_sd = donor_model.state_dict()

    # Patch base with donor
    patched_sd, meta = patch_state_dict(base_sd, donor_sd, "token_embed")

    # Verify patch worked
    assert not torch.allclose(base_sd["token_embed.weight"], patched_sd["token_embed.weight"])
    assert torch.allclose(donor_sd["token_embed.weight"], patched_sd["token_embed.weight"])
    assert "token_embed.weight" in meta["patched_keys"]

    # Restore original weights
    patched_model = DummyModel()
    patched_model.load_state_dict(patched_sd)

    restored_sd, _ = patch_state_dict(patched_sd, base_sd, "token_embed")

    # Verify it matches original base
    assert torch.allclose(base_sd["token_embed.weight"], restored_sd["token_embed.weight"])

def test_patch_state_dict_fractional_mlp():
    base_sd = {
        "transformer.layers.0.linear1.weight": torch.zeros(32, 16),
        "transformer.layers.0.linear1.bias": torch.zeros(32),
        "transformer.layers.0.linear2.weight": torch.zeros(16, 32),
        "transformer.layers.0.linear2.bias": torch.zeros(16),
    }
    donor_sd = {
        "transformer.layers.0.linear1.weight": torch.ones(32, 16),
        "transformer.layers.0.linear1.bias": torch.ones(32),
        "transformer.layers.0.linear2.weight": torch.ones(16, 32),
        "transformer.layers.0.linear2.bias": torch.ones(16),
    }

    patched_sd, meta = patch_state_dict_fractional(
        base_sd, donor_sd, "mlp", fraction=0.5, d_model=16, d_ff=32, seed=42
    )

    assert meta["fraction"] == 0.5

    # 50% of 32 neurons = 16 neurons
    assert patched_sd["transformer.layers.0.linear1.weight"].sum().item() == 16 * 16
    assert patched_sd["transformer.layers.0.linear1.bias"].sum().item() == 16
    assert patched_sd["transformer.layers.0.linear2.weight"].sum().item() == 16 * 16

def test_patch_state_dict_fractional_attn():
    d_model = 16
    n_heads = 4
    head_dim = d_model // n_heads

    base_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(3 * d_model, d_model),
        "transformer.layers.0.self_attn.in_proj_bias": torch.zeros(3 * d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(d_model, d_model),
        "transformer.layers.0.self_attn.out_proj.bias": torch.zeros(d_model),
    }

    donor_sd = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(3 * d_model, d_model),
        "transformer.layers.0.self_attn.in_proj_bias": torch.ones(3 * d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.ones(d_model, d_model),
        "transformer.layers.0.self_attn.out_proj.bias": torch.ones(d_model),
    }

    patched_sd, meta = patch_state_dict_fractional(
        base_sd, donor_sd, "attn", fraction=0.25, n_heads=n_heads, d_model=d_model, seed=42
    )

    assert meta["fraction"] == 0.25

    # 25% of 4 heads = 1 head
    # in_proj_weight is [3 * d_model, d_model], patched 1 head per Q, K, V (1 * head_dim * d_model * 3)
    assert patched_sd["transformer.layers.0.self_attn.in_proj_weight"].sum().item() == 3 * head_dim * d_model
    assert patched_sd["transformer.layers.0.self_attn.in_proj_bias"].sum().item() == 3 * head_dim
    # out_proj.weight is [d_model, d_model], patched 1 head (d_model * head_dim)
    assert patched_sd["transformer.layers.0.self_attn.out_proj.weight"].sum().item() == d_model * head_dim


def test_matrix_csv_schema():
    import pandas as pd
    from src.transplant.run_transplants import patch_state_dict

    # We will simulate the matrix generation without running the model
    base_model = DummyModel()
    donor_model = DummyModel()

    results = []

    for comp in ["token_embed"]:
        patched_sd, meta = patch_state_dict(base_model.state_dict(), donor_model.state_dict(), comp)

        row = {
            "component": comp,
            "donor": "pure",
            "recipient": "contam",
            "zero_shot_acc": 0.95,
            "donor_hash": meta["donor_hash"],
            "recipient_hash": meta["base_hash"],
            "patched_keys": ",".join(meta["patched_keys"]),
            "seed": 42,
        }
        results.append(row)

    df = pd.DataFrame(results)

    expected_columns = [
        "component", "donor", "recipient", "zero_shot_acc",
        "donor_hash", "recipient_hash", "patched_keys", "seed"
    ]

    for col in expected_columns:
        assert col in df.columns

    assert df["component"].iloc[0] == "token_embed"
    assert df["donor"].iloc[0] == "pure"
    assert "token_embed.weight" in df["patched_keys"].iloc[0]
