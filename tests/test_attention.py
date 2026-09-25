import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import os
import tempfile

from src.attention.extraction import compute_entropy, extract_attention
from src.attention.visualization import plot_attention_heatmaps, plot_attention_trajectory, plot_attention_diff
from src.model import ModularArithmeticTransformer

def test_compute_entropy():
    # 2 tokens, deterministic attention (one head)
    # Head 0: attends 100% to token 0
    attn = np.array([
        [[1.0, 0.0],
         [1.0, 0.0]]
    ])

    # Entropy should be 0 because probabilities are 1 and 0 (log(1)=0, log(0)->clamped)
    entropy = compute_entropy(attn)
    assert np.isclose(entropy, 0.0, atol=1e-4)

    # Head 0: attends uniformly (0.5, 0.5)
    attn_uniform = np.array([
        [[0.5, 0.5],
         [0.5, 0.5]]
    ])

    # - (0.5 * log(0.5) + 0.5 * log(0.5)) = -log(0.5) = 0.693147
    entropy_uniform = compute_entropy(attn_uniform)
    assert np.isclose(entropy_uniform, 0.693147, atol=1e-4)

def test_compute_entropy_with_mask():
    attn = np.array([
        [[0.5, 0.5],
         [0.5, 0.5]]
    ])
    # Mask out the second token
    mask = np.array([
        [[True, False],
         [True, False]]
    ])

    # After masking and renormalizing, it should be [[1.0, 0.0], [1.0, 0.0]]
    entropy = compute_entropy(attn, mask=mask)
    assert np.isclose(entropy, 0.0, atol=1e-4)

def test_extract_attention():
    model = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2, n_layers=1)
    model.eval()

    # Batch size 1, seq_len 2
    x = torch.tensor([[3, 5]], dtype=torch.long)

    with extract_attention(model) as extracted:
        _ = model(x)

    assert len(extracted) == 1
    assert 0 in extracted[0] # Layer 0

    attn = extracted[0][0]
    assert attn.shape == (1, 2, 2, 2) # batch, heads, seq, seq

    # Check that it sums to 1 along the last axis
    assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-5)

def test_visualization_heatmaps(tmp_path):
    attn = np.random.rand(4, 2, 2)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    tokens = ["a", "b"]
    save_path = tmp_path / "heatmap.png"

    plot_attention_heatmaps(attn, tokens, 0, str(save_path))
    assert save_path.exists()

def test_visualization_trajectory(tmp_path):
    steps = [100, 200, 300]
    entropies = [0.6, 0.4, 0.1]
    grokking_step = 250

    save_path = tmp_path / "trajectory.png"

    plot_attention_trajectory(steps, entropies, grokking_step, str(save_path))
    assert save_path.exists()

def test_visualization_diff(tmp_path):
    attn_a = np.random.rand(4, 2, 2)
    attn_b = np.random.rand(4, 2, 2)

    tokens = ["a", "b"]
    save_path = tmp_path / "diff.png"

    plot_attention_diff(attn_a, attn_b, tokens, 0, str(save_path))
    assert save_path.exists()

def test_visualization_critical_token_trajectory(tmp_path):
    from src.attention.visualization import plot_critical_token_trajectory
    steps = [100, 200, 300]
    attn_to_critical = np.random.rand(3, 4) # 3 steps, 4 heads
    tokens = ["a", "b"]
    grokking_step = 250

    save_path = tmp_path / "critical_trajectory.png"
    plot_critical_token_trajectory(steps, attn_to_critical, 0, tokens, grokking_step, str(save_path))
    assert save_path.exists()

def test_extract_attention_no_checkpoints():
    from src.attention_visualizer import get_checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpts = get_checkpoints(tmpdir)
        assert len(ckpts) == 0

def test_load_model_malformed_checkpoint():
    from src.attention_visualizer import load_model
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "bad.pt")
        with open(ckpt_path, "wb") as f:
            f.write(b"not a valid checkpoint")
        with pytest.raises(Exception):
            load_model(ckpt_path, torch.device("cpu"))
