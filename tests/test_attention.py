import pytest
import torch
import numpy as np
from src.model import ModularArithmeticTransformer
from src.analysis.attention import (
    extract_attention_weights,
    compute_entropy,
    compute_positional_concentration,
    compute_head_similarity,
    analyze_attention
)
import matplotlib
matplotlib.use('Agg')
from src.viz.attention import plot_attention_grid, plot_attention_diff_grid


@pytest.fixture
def small_model():
    """Create a very small model for testing."""
    return ModularArithmeticTransformer(
        prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=1
    )


@pytest.fixture
def dummy_inputs():
    """Create dummy inputs for the small model."""
    return torch.randint(0, 5, (4, 2))  # batch=4, seq=2


def test_extract_attention_weights(small_model, dummy_inputs):
    """Test extracting attention weights from the model."""
    weights = extract_attention_weights(small_model, dummy_inputs)

    # Expected shape: (n_layers, n_heads, seq_len, seq_len)
    assert weights.shape == (1, 2, 2, 2)
    # Weights should sum to 1 along the last dimension
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 2, 2))


def test_compute_entropy():
    """Test entropy computation with known values."""
    # Create a uniform distribution
    attn_uniform = torch.ones(1, 1, 2, 2) * 0.5
    entropy = compute_entropy(attn_uniform)
    # - (0.5 * log(0.5) + 0.5 * log(0.5)) = -log(0.5) = log(2)
    expected_entropy = torch.ones(1, 1, 2) * torch.log(torch.tensor(2.0))
    assert torch.allclose(entropy, expected_entropy, atol=1e-5)

    # Create a one-hot distribution (zero entropy)
    attn_onehot = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    entropy_onehot = compute_entropy(attn_onehot)
    expected_zero = torch.zeros(1, 1, 2)
    assert torch.allclose(entropy_onehot, expected_zero, atol=1e-5)


def test_compute_positional_concentration():
    """Test positional concentration computation."""
    attn = torch.tensor([[[[0.8, 0.2], [0.3, 0.7]]]])
    concentration = compute_positional_concentration(attn)
    expected = torch.tensor([[[0.8, 0.7]]])
    assert torch.allclose(concentration, expected)


def test_compute_head_similarity():
    """Test head similarity matrix computation."""
    # Two layers, 1 head each, 2x2 sequence
    attn1 = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    attn2 = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    # Concatenate along layer dim to get (2, 1, 2, 2)
    attn = torch.cat([attn1, attn2], dim=0)

    sim = compute_head_similarity(attn)
    # Expected shape: (2*1, 2*1) = (2, 2)
    assert sim.shape == (2, 2)

    # Self similarity should be 1
    assert torch.allclose(torch.diag(sim), torch.ones(2))

    # The two maps are orthogonal, similarity should be 0
    assert torch.allclose(sim[0, 1], torch.tensor(0.0))
    assert torch.allclose(sim[1, 0], torch.tensor(0.0))


def test_analyze_attention(small_model, dummy_inputs):
    """Test full analysis pipeline."""
    weights = extract_attention_weights(small_model, dummy_inputs)
    results = analyze_attention(weights)

    assert "entropy" in results
    assert "mean_entropy" in results
    assert "concentration" in results
    assert "mean_concentration" in results
    assert "head_similarity" in results

    assert isinstance(results["entropy"], np.ndarray)
    assert results["entropy"].shape == (1, 2, 2)
    assert results["mean_entropy"].shape == (1, 2)
    assert results["head_similarity"].shape == (2, 2)


def test_plot_attention_grid(tmp_path):
    """Test plotting attention grid."""
    attn = torch.rand(2, 3, 4, 4)
    # Normalize
    attn = attn / attn.sum(dim=-1, keepdim=True)

    out_file = tmp_path / "test_grid.png"
    fig = plot_attention_grid(attn, output_path=out_file)

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_attention_diff_grid(tmp_path):
    """Test plotting attention diff grid."""
    attn_a = torch.rand(2, 3, 4, 4)
    attn_b = torch.rand(2, 3, 4, 4)

    out_file = tmp_path / "test_diff.png"
    fig = plot_attention_diff_grid(attn_a, attn_b, output_path=out_file)

    assert out_file.exists()
    assert out_file.stat().st_size > 0
