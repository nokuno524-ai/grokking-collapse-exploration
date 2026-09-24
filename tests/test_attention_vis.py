"""
Tests for attention visualization tools.
"""

import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model import ModularArithmeticTransformer
from src.attention_extraction import extract_attention, compute_entropy
from scripts.visualize_attention import get_checkpoints, load_model_from_checkpoint, render_attention_heatmaps, render_head_similarity, render_attention_entropy


def test_compute_entropy():
    """Test compute_entropy function."""
    # Uniform distribution
    w = np.array([0.5, 0.5])
    e = compute_entropy(w)
    assert np.isclose(e, -np.log(0.5))

    # Delta distribution
    w = np.array([1.0, 0.0])
    e = compute_entropy(w)
    assert np.isclose(e, 0.0)

    # Empty distribution
    w = np.array([0.0, 0.0])
    e = compute_entropy(w)
    assert np.isclose(e, 0.0)


def test_extraction_shapes():
    """Test extract_attention context manager extraction shapes."""
    torch.manual_seed(42)
    model = ModularArithmeticTransformer(
        prime=59, d_model=16, n_heads=2, d_ff=32, n_layers=1
    )
    model.eval()

    inputs = torch.randint(0, 59, (4, 2))  # batch=4, seq_len=2

    with extract_attention(model) as extracted_maps:
        with torch.no_grad():
            out = model(inputs)

    assert len(extracted_maps) == 1
    # Shape: [batch_size, num_heads, seq_len, seq_len]
    assert extracted_maps[0].shape == (4, 2, 2, 2)


def test_determinism_fixed_seeds():
    """Test determinism of attention extraction with fixed seeds."""
    torch.manual_seed(42)
    model1 = ModularArithmeticTransformer(
        prime=59, d_model=16, n_heads=2, d_ff=32, n_layers=1
    )
    model1.eval()

    torch.manual_seed(42)
    model2 = ModularArithmeticTransformer(
        prime=59, d_model=16, n_heads=2, d_ff=32, n_layers=1
    )
    model2.eval()

    inputs = torch.randint(0, 59, (4, 2))

    with extract_attention(model1) as extracted_maps1:
        with torch.no_grad():
            model1(inputs)

    with extract_attention(model2) as extracted_maps2:
        with torch.no_grad():
            model2(inputs)

    assert np.allclose(extracted_maps1[0], extracted_maps2[0])


def test_missing_checkpoint(tmp_path):
    """Test handling of missing checkpoints."""
    # Empty directory
    checkpoints = get_checkpoints(tmp_path)
    assert len(checkpoints) == 0

    # Create non-checkpoint file
    (tmp_path / "results.json").touch()
    checkpoints = get_checkpoints(tmp_path)
    assert len(checkpoints) == 0


@patch("scripts.visualize_attention.plt.savefig")
def test_figure_creation(mock_savefig, tmp_path):
    """Test that rendering functions create files and JSON correctly (mocking plt.savefig)."""
    # Create fake attention maps: 2 checkpoints
    attns1 = np.random.rand(4, 2, 2, 2)  # [batch, heads, seq, seq]
    attns2 = np.random.rand(4, 2, 2, 2)

    # Normalize along last dimension to make them valid attention distributions
    attns1 = attns1 / attns1.sum(axis=-1, keepdims=True)
    attns2 = attns2 / attns2.sum(axis=-1, keepdims=True)

    all_attns = [(100, attns1), (200, attns2)]

    # Test heatmap rendering
    render_attention_heatmaps(attns1, 100, tmp_path)
    assert mock_savefig.call_count >= 1
    # Extract the filename from the first positional argument of the first call
    # In plt.savefig, the first arg is typically the filename
    called_path = mock_savefig.call_args[0][0]
    assert "attention_heatmaps_step_100.png" in str(called_path)

    # Reset mock
    mock_savefig.reset_mock()

    # Test head similarity
    render_head_similarity(all_attns, tmp_path)
    assert mock_savefig.call_count >= 1
    called_path = mock_savefig.call_args[0][0]
    assert "head_similarity_over_time.png" in str(called_path)

    # Reset mock
    mock_savefig.reset_mock()

    # Test attention entropy
    render_attention_entropy(all_attns, tmp_path)
    assert mock_savefig.call_count >= 1
    called_path = mock_savefig.call_args[0][0]
    assert "attention_entropy_over_time.png" in str(called_path)

    # Verify JSON was written
    json_path = tmp_path / "attention_entropy.json"
    assert json_path.exists()

    with open(json_path, "r") as f:
        data = json.load(f)

    assert "steps" in data
    assert data["steps"] == [100, 200]
    assert "head_0" in data
    assert "head_1" in data
    assert len(data["head_0"]) == 2
