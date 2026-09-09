import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer
from src.transplant.alignment import align_mlp_layer, align_attention_heads, align_models
import copy

def test_align_mlp_layer():
    torch.manual_seed(42)
    d_model, d_ff = 4, 8
    w1 = torch.randn(d_ff, d_model)
    w2 = torch.randn(d_model, d_ff)
    b1 = torch.randn(d_ff)

    act_a = torch.randn(10, d_ff)

    # create a permutation
    perm = torch.randperm(d_ff)

    w1_perm = w1[perm, :]
    w2_perm = w2[:, perm]
    b1_perm = b1[perm]

    act_b = act_a[:, perm]

    # The alignment should recover this permutation and bring the permuted weights back to w1
    p_w1, p_w2, p_b1, sim = align_mlp_layer(w1, w1_perm, w2, w2_perm, b1, b1_perm, act_a, act_b)

    assert torch.allclose(p_w1, w1, atol=1e-5), "MLP W1 alignment failed"
    assert torch.allclose(p_w2, w2, atol=1e-5), "MLP W2 alignment failed"
    assert torch.allclose(p_b1, b1, atol=1e-5), "MLP B1 alignment failed"

def test_align_models():
    torch.manual_seed(42)
    model_a = ModularArithmeticTransformer(d_model=16, d_ff=32, n_heads=2)

    # We create model_b by copying model_a and permuting its first layer's MLP and Attention
    model_b = copy.deepcopy(model_a)

    # Permute MLP
    d_ff = 32
    perm_mlp = torch.randperm(d_ff)
    with torch.no_grad():
        w1 = model_b.transformer.layers[0].linear1.weight
        w2 = model_b.transformer.layers[0].linear2.weight
        b1 = model_b.transformer.layers[0].linear1.bias

        w1.copy_(w1[perm_mlp, :])
        w2.copy_(w2[:, perm_mlp])
        b1.copy_(b1[perm_mlp])

        # Permute Attention
        d_model = 16
        n_heads = 2
        head_dim = d_model // n_heads

        perm_heads = torch.randperm(n_heads)

        in_proj = model_b.transformer.layers[0].self_attn.in_proj_weight
        out_proj = model_b.transformer.layers[0].self_attn.out_proj.weight

        # Manually permuting heads in qkv
        q, k, v = in_proj.chunk(3, dim=0)
        def permute_qkv(w):
            w = w.view(n_heads, head_dim, d_model)
            w = w[perm_heads, :, :]
            return w.view(d_model, d_model)

        model_b.transformer.layers[0].self_attn.in_proj_weight.copy_(
            torch.cat([permute_qkv(q), permute_qkv(k), permute_qkv(v)], dim=0)
        )

        op = out_proj.view(d_model, n_heads, head_dim)
        op = op[:, perm_heads, :]
        model_b.transformer.layers[0].self_attn.out_proj.weight.copy_(
            op.view(d_model, d_model)
        )

    aligned_b, sim_before, sim_after = align_models(model_a, model_b)

    # Should restore output equivalence
    x = torch.randint(0, 59, (10, 2))

    out_a = model_a(x)
    out_aligned_b = aligned_b(x)

    assert torch.allclose(out_a, out_aligned_b, atol=1e-5), "Aligned model does not match original outputs"

import os
from pathlib import Path
import json

def test_atlas_aggregation(tmp_path):
    from src.transplant.atlas import main as atlas_main
    import sys

    pure_dir = tmp_path / "dummy_pure"
    contam_dir = tmp_path / "dummy_contam"
    out_dir = tmp_path / "atlas_out"

    for d, is_pure in [(pure_dir, True), (contam_dir, False)]:
        d.mkdir(parents=True, exist_ok=True)
        cfg = {
            "prime": 59,
            "d_model": 128,
            "n_heads": 4,
            "d_ff": 512,
            "n_layers": 1,
            "train_fraction": 0.3,
            "collapse_level": 0.0 if is_pure else 0.5,
            "collapse_severity": 0.5,
            "noise_fraction": 0.0,
            "seed": 42
        }
        with open(d / "results.json", "w") as f:
            json.dump({"config": cfg}, f)
        model = ModularArithmeticTransformer(**{k: cfg[k] for k in ["prime", "d_model", "n_heads", "d_ff", "n_layers"]})
        torch.save({"model_state": model.state_dict(), "config": cfg}, d / "checkpoint_1000.pt")

    old_argv = sys.argv
    sys.argv = ["atlas.py", "--pure-run", str(pure_dir), "--contam-run", str(contam_dir), "--output-dir", str(out_dir)]

    try:
        atlas_main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "atlas_results.csv").exists(), "Atlas did not produce CSV"
    assert (out_dir / "atlas_pure_to_contam.png").exists(), "Atlas did not produce heatmap"

    import pandas as pd
    df = pd.read_csv(out_dir / "atlas_results.csv")
    assert len(df) > 0, "Atlas CSV is empty"
    assert "direction" in df.columns
    assert "component" in df.columns
    assert "test_acc" in df.columns
    assert "effect_size" in df.columns
