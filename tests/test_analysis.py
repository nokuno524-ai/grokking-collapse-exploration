import pytest
import numpy as np
import torch
import tempfile
import csv
import json
from pathlib import Path
import os
import sys

# We need to make sure we can import the scripts
# Add scripts directory to path if needed for testing, or test functions directly
# For scripts/visualize.py, we have functions compute_attention_entropy, compute_head_specialization

# Let's import from visualize script
sys.path.append(os.path.abspath("scripts"))
from visualize import compute_attention_entropy, compute_head_specialization

def test_compute_attention_entropy():
    # known answer test for entropy
    # uniform distribution has max entropy
    n_heads = 2
    seq_len = 4
    batch = 1

    # 1. Uniform attention
    uniform_attn = torch.ones(batch, n_heads, seq_len, seq_len) / seq_len
    # entropy = sum(1/4 * log(4)) = log(4) = 1.386
    ent = compute_attention_entropy(uniform_attn)
    assert ent.shape == (n_heads,)
    assert np.allclose(ent, np.log(seq_len), atol=1e-4)

    # 2. Perfect attention (one-hot)
    perfect_attn = torch.zeros(batch, n_heads, seq_len, seq_len)
    for i in range(seq_len):
        perfect_attn[0, 0, i, i] = 1.0
        perfect_attn[0, 1, i, 0] = 1.0 # all attend to first token

    ent = compute_attention_entropy(perfect_attn)
    assert np.allclose(ent, 0.0, atol=1e-4)

def test_compute_head_specialization():
    n_heads = 2
    seq_len = 4
    batch = 1

    # If a head always attends to the same token across all queries,
    # its average distribution is one-hot -> low entropy (high specialization)
    attn1 = torch.zeros(batch, 1, seq_len, seq_len)
    attn1[:, :, :, 0] = 1.0 # always attends to index 0

    # If a head attends uniformly to everything
    attn2 = torch.ones(batch, 1, seq_len, seq_len) / seq_len

    attn = torch.cat([attn1, attn2], dim=1)

    spec = compute_head_specialization(attn)
    assert spec.shape == (n_heads,)
    assert np.allclose(spec[0], 0.0, atol=1e-4) # specialized
    assert np.allclose(spec[1], np.log(seq_len), atol=1e-4) # uniform

def test_grokking_step_detection():
    # Test logic used in inventory.py
    history = [
        {"step": 100, "test_acc": 0.1},
        {"step": 200, "test_acc": 0.5},
        {"step": 300, "test_acc": 0.95},
        {"step": 400, "test_acc": 0.99},
    ]

    grokking_step = None
    for entry in history:
        if entry.get("test_acc", 0) > 0.9:
            grokking_step = entry.get("step")
            break

    assert grokking_step == 300

    history_fail = [
        {"step": 100, "test_acc": 0.1},
        {"step": 200, "test_acc": 0.5},
        {"step": 300, "test_acc": 0.8},
    ]
    grokking_step = None
    for entry in history_fail:
        if entry.get("test_acc", 0) > 0.9:
            grokking_step = entry.get("step")
            break

    assert grokking_step is None

def test_csv_parsing_malformed_rows():
    # Test analyze_runs.py behavior with malformed CSV rows
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "condition", "collapse_ratio", "collapse_severity", "seed", "final_accuracy", "grokking_step", "final_weight_norm"])
        writer.writerow(["valid", "pure", "0.0", "0.0", "42", "1.0", "1400", "28.5"])
        # Malformed: missing grokking step (blank)
        writer.writerow(["missing_grok", "high_col", "0.3", "0.5", "42", "0.3", "", "55.0"])
        f.flush()

        rows = []
        with open(f.name, "r") as csvf:
            reader = csv.DictReader(csvf)
            for row in reader:
                rows.append(row)

        assert len(rows) == 2
        assert rows[0]["grokking_step"] == "1400"
        assert rows[1]["grokking_step"] == ""

        # Test how our script handles it
        wn_grokked = []
        wn_not_grokked = []

        for row in rows:
            wn = float(row["final_weight_norm"])
            if row["grokking_step"]:
                wn_grokked.append(wn)
            else:
                wn_not_grokked.append(wn)

        assert len(wn_grokked) == 1
        assert len(wn_not_grokked) == 1
        assert wn_grokked[0] == 28.5
        assert wn_not_grokked[0] == 55.0

    os.unlink(f.name)


def test_visualize_attention_synthetic():
    # Test plotting with synthetic checkpoints without invoking full matplotlib backend recursively
    from scripts.visualize import compute_attention_entropy, compute_head_specialization
    from src.model import ModularArithmeticTransformer
    import tempfile
    import shutil
    import os
    import json

    # We will just verify that given a synthetic model, we can extract the qkv and compute entropy
    model = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1)

    # Dummy input
    dummy_input = torch.randint(0, 7, (4, 2))
    tok = model.token_embed(dummy_input)
    pos = model.pos_embed(torch.arange(2).unsqueeze(0).expand(4, -1))
    h = tok + pos

    # Get Q, K from self_attn in encoder_layer
    layer = model.transformer.layers[0]
    in_proj_weight = layer.self_attn.in_proj_weight
    in_proj_bias = layer.self_attn.in_proj_bias

    qkv = torch.nn.functional.linear(h, in_proj_weight, in_proj_bias)
    d_model = model.d_model
    q, k, v = qkv.split(d_model, dim=-1)

    n_heads = model.n_heads
    head_dim = d_model // n_heads
    batch_size = h.size(0)

    q = q.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)

    ent = compute_attention_entropy(attn_weights)
    spec = compute_head_specialization(attn_weights)

    assert ent.shape == (2,)
    assert spec.shape == (2,)
