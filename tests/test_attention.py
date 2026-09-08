import os
import torch
import numpy as np
import pytest
from src.model import ModularArithmeticTransformer
from src.analysis.attention import calculate_attention_entropy, compute_js_divergence, compute_head_similarity
from src.viz.attention import plot_attention_heatmaps, plot_entropy_trajectories, plot_head_clustering

def test_extraction_hook():
    """Test that patch_self_attn captures correct shapes and layer indices."""
    # We must patch from the script logic
    from scripts.extract_attention import patch_self_attn

    model = ModularArithmeticTransformer(n_layers=2, n_heads=2)
    weights_list = patch_self_attn(model)

    x = torch.randint(0, 59, (4, 2))
    _ = model(x)

    assert len(weights_list) == 2, "Should capture weights from 2 layers"

    layer_0_idx, layer_0_w = weights_list[0]
    assert layer_0_idx == 0
    # Shape: (batch, heads, seq, seq) -> (4, 2, 2, 2)
    assert layer_0_w.shape == (4, 2, 2, 2)

    layer_1_idx, layer_1_w = weights_list[1]
    assert layer_1_idx == 1
    assert layer_1_w.shape == (4, 2, 2, 2)

def test_calculate_attention_entropy():
    """Test entropy calculation with uniform and one-hot distributions."""
    # (batch=1, heads=1, seq=1, seq=2)
    # Uniform: [0.5, 0.5] -> entropy = -(0.5*ln(0.5) + 0.5*ln(0.5)) = ln(2) ~ 0.693
    uniform_w = np.array([[[[0.5, 0.5]]]])
    ent_uniform = calculate_attention_entropy(uniform_w)
    assert np.allclose(ent_uniform, np.log(2)), f"Expected ~0.693, got {ent_uniform}"

    # One-hot: [1.0, 0.0] -> entropy = -(1*ln(1) + 0) = 0
    onehot_w = np.array([[[[1.0, 0.0]]]])
    ent_onehot = calculate_attention_entropy(onehot_w)
    assert np.allclose(ent_onehot, 0.0, atol=1e-5), f"Expected ~0, got {ent_onehot}"

def test_compute_js_divergence():
    """Test JS divergence known values."""
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert np.isclose(compute_js_divergence(p, q), 0.0)

    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    # JSD between mutually exclusive is ln(2) ~ 0.693
    assert np.isclose(compute_js_divergence(p, q), np.log(2))

def test_compute_head_similarity():
    """Test head similarity matrix construction."""
    # 2 heads. Head 0 and Head 1 have identical patterns.
    w = np.array([[
        [[1.0, 0.0], [0.5, 0.5]], # Head 0
        [[1.0, 0.0], [0.5, 0.5]]  # Head 1
    ]])

    sim = compute_head_similarity(w)
    assert sim.shape == (2, 2)
    assert np.allclose(sim, np.ones((2, 2))) # All similarities should be 1.0

    # Dissimilar heads
    w2 = np.array([[
        [[1.0, 0.0], [1.0, 0.0]], # Head 0
        [[0.0, 1.0], [0.0, 1.0]]  # Head 1
    ]])
    sim2 = compute_head_similarity(w2)
    # Off-diagonal should be 0 since they are mutually exclusive (JSD = ln(2), Sim = 1 - ln(2)/ln(2) = 0)
    assert np.isclose(sim2[0, 1], 0.0)

def test_visualization_smoke(tmp_path):
    """Smoke test plotting functions to ensure they don't crash."""
    out_dir = str(tmp_path)

    # Heatmap
    attn = np.random.rand(2, 4, 4)
    # Normalize to simulate attention
    attn = attn / attn.sum(axis=-1, keepdims=True)
    heatmap_out = os.path.join(out_dir, "heatmap.png")
    plot_attention_heatmaps(attn, layer_idx=0, output_path=heatmap_out)
    assert os.path.exists(heatmap_out)

    # Entropy
    steps = [0, 100, 200]
    data = {
        "pure": {"head_0": np.array([1.0, 0.5, 0.1]), "head_1": np.array([1.0, 0.8, 0.5])},
        "contam": {"head_0": np.array([1.0, 0.9, 0.9]), "head_1": np.array([1.0, 1.0, 1.0])}
    }
    entropy_out = os.path.join(out_dir, "entropy.png")
    plot_entropy_trajectories(steps, data, entropy_out)
    assert os.path.exists(entropy_out)

    # Clustering
    sim_matrix = np.array([
        [1.0, 0.9, 0.1, 0.2],
        [0.9, 1.0, 0.15, 0.1],
        [0.1, 0.15, 1.0, 0.8],
        [0.2, 0.1, 0.8, 1.0]
    ])
    labels = ["H0", "H1", "H2", "H3"]
    cluster_out = os.path.join(out_dir, "cluster.png")
    plot_head_clustering(sim_matrix, labels, cluster_out)
    assert os.path.exists(cluster_out)
