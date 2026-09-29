import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Ensure src is in path for testing
sys.path.insert(0, os.path.abspath('.'))

from src.attention.visualizer import (
    plot_attention_grid,
    plot_attention_entropy,
    plot_head_specialization_clusters,
    animate_attention_evolution
)
from src.attention.dynamics import compute_shannon_entropy


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def synthetic_attention_tensor():
    # shape: (n_layers, n_heads, seq_len, seq_len)
    # 2 layers, 4 heads, seq_len 10
    tensor = np.random.rand(2, 4, 10, 10)
    # Normalize rows to sum to 1 to represent valid attention
    tensor = tensor / tensor.sum(axis=-1, keepdims=True)
    return tensor


def test_plot_attention_grid(synthetic_attention_tensor, temp_dir):
    out_path = temp_dir / "grid.png"
    plot_attention_grid(synthetic_attention_tensor, out_path)
    assert out_path.exists(), "Plot file was not created"


def test_plot_attention_entropy(temp_dir):
    steps = np.arange(10)
    entropy = np.random.rand(10, 4)  # 10 steps, 4 heads

    out_path = temp_dir / "entropy.png"
    plot_attention_entropy(steps, entropy, out_path, labels=["H1", "H2", "H3", "H4"])
    assert out_path.exists(), "Entropy plot file was not created"


def test_plot_head_specialization_clusters(synthetic_attention_tensor, temp_dir):
    out_path = temp_dir / "clusters.png"
    plot_head_specialization_clusters(synthetic_attention_tensor, out_path)
    assert out_path.exists(), "Clustering plot file was not created"


def test_plot_head_specialization_clusters_single_head(temp_dir):
    # Edge case: only 1 head. pdist will fail if not handled properly.
    tensor = np.random.rand(1, 1, 10, 10)
    out_path = temp_dir / "clusters_single.png"
    plot_head_specialization_clusters(tensor, out_path)
    assert out_path.exists(), "Single head clustering plot should still create a file"


def test_animate_attention_evolution(temp_dir):
    # shape: (steps, seq_len, seq_len)
    history = np.random.rand(5, 10, 10)
    steps = np.arange(5) * 10

    out_path = temp_dir / "anim.gif"
    animate_attention_evolution(history, out_path, steps=steps, fps=2)
    assert out_path.exists(), "Animation file was not created"


def test_compute_shannon_entropy(synthetic_attention_tensor):
    entropy = compute_shannon_entropy(synthetic_attention_tensor)
    assert entropy.shape == (2, 4)
    # Max entropy for 10 choices is log2(10) ~ 3.32
    assert np.all(entropy >= 0) and np.all(entropy <= np.log2(10) + 1e-5)

    # Test identical degenerate distribution
    degenerate = np.zeros((1, 1, 10, 10))
    degenerate[:, :, :, 0] = 1.0  # All weight on first token
    deg_entropy = compute_shannon_entropy(degenerate)
    assert np.allclose(deg_entropy, 0.0, atol=1e-5)
