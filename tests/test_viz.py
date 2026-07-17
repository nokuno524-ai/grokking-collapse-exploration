import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModularArithmeticTransformer
from viz.attention_patterns import extract_attention_weights
from viz.weight_norms import get_layer_norms
from viz.loss_landscape import get_random_directions
from viz.grokking_cliff import plot_phase_diagram

def test_extract_attention_weights():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2)
    inputs = torch.randint(0, 59, (4, 2))
    attn_weights = extract_attention_weights(model, inputs)

    assert attn_weights is not None
    # shape: (batch_size, num_heads, tgt_len, src_len)
    assert attn_weights.shape == (4, 2, 2, 2)

    # Check if they are valid probabilities (sum to 1)
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))

def test_get_layer_norms():
    model = ModularArithmeticTransformer(prime=59, d_model=32)
    state_dict = model.state_dict()
    norms = get_layer_norms(state_dict)

    assert "token_embed" in norms
    assert "pos_embed" in norms
    assert "layer_0_attn_in" in norms
    assert "layer_0_attn_out" in norms
    assert "output_head" in norms

    for k, v in norms.items():
        assert isinstance(v, float)
        assert v > 0

def test_get_random_directions():
    model = ModularArithmeticTransformer(prime=59, d_model=32)
    d1, d2 = get_random_directions(model)

    assert len(d1) == len(list(model.parameters()))
    assert len(d2) == len(list(model.parameters()))

    # Check orthogonality
    dot_product = sum((x * y).sum().item() for x, y in zip(d1, d2))
    assert abs(dot_product) < 1e-5

def test_plot_phase_diagram():
    # Mock data
    df = pd.DataFrame({
        "level": [0.0, 0.0, 0.1, 0.1],
        "severity": [0.5, 0.9, 0.5, 0.9],
        "test_acc": [1.0, 1.0, 0.8, 0.5],
        "grokking_step": [1000, 2000, 5000, 10000]
    })

    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_phase.png"
        plot_phase_diagram(df, "test_acc", "Test", "viridis", save_path)
        assert save_path.exists()
