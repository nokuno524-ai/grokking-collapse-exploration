import pytest
import torch
import numpy as np
from pathlib import Path

from src.model import ModularArithmeticTransformer
from src.analysis.attention_analysis import extract_attention_weights, load_model_from_checkpoint
from src.analysis.phase_detector import detect_grokking_transition, detect_fourier_shift


def test_extract_attention_weights_shape():
    """Test that extract_attention_weights returns the correct shape."""
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4, d_ff=512, n_layers=1)
    model.eval()

    batch_size = 10
    x = torch.randint(0, 59, (batch_size, 2))

    attn_weights = extract_attention_weights(model, x)

    # Expected shape: (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == (batch_size, 4, 2, 2)

    # Values should be between 0 and 1, and sum to 1 over the last dimension
    assert torch.all(attn_weights >= 0)
    assert torch.all(attn_weights <= 1)

    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_load_model_from_checkpoint(tmp_path):
    """Test loading a model from a checkpoint dictionary format."""
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4, d_ff=512, n_layers=1)
    ckpt_path = tmp_path / "checkpoint_100.pt"
    torch.save({
        "step": 100,
        "model_state": model.state_dict(),
        "config": {
            "prime": 59,
            "d_model": 128,
            "n_heads": 4,
            "d_ff": 512,
            "n_layers": 1
        }
    }, ckpt_path)

    loaded_model = load_model_from_checkpoint(ckpt_path)
    assert isinstance(loaded_model, ModularArithmeticTransformer)
    assert loaded_model.prime == 59
    assert loaded_model.d_model == 128


def test_detect_grokking_transition():
    """Test grokking transition detection with synthetic data."""
    steps = list(range(0, 1000, 10))
    # 100 steps total

    # Case 1: No grokking
    metrics = [0.1] * 100
    assert detect_grokking_transition(metrics, steps, threshold=0.9, window_size=10) is None

    # Case 2: Grokking but not sustained enough (window=10)
    metrics = [0.1] * 50 + [0.95] * 5 + [0.1] * 45
    assert detect_grokking_transition(metrics, steps, threshold=0.9, window_size=10) is None

    # Case 3: Proper grokking
    metrics = [0.1] * 50 + [0.95] * 50
    # The transition starts at index 50, which is step 500
    res = detect_grokking_transition(metrics, steps, threshold=0.9, window_size=10)
    assert res is not None
    assert res["step"] == 500
    assert "confidence_metrics" in res
    assert res["confidence_metrics"]["stability_score"] == 1.0


def test_detect_fourier_shift():
    """Test fourier shift detection with synthetic data."""
    steps = list(range(0, 1000, 10))

    # Case 1: Always uniform
    conc = [0.05] * 100
    res = detect_fourier_shift(conc, steps, uniform_threshold=0.1, concentrated_threshold=0.5, window_size=10)
    assert not res["is_shifted"]

    # Case 2: Proper shift
    conc = [0.05] * 40 + [0.3] * 20 + [0.8] * 40
    res = detect_fourier_shift(conc, steps, uniform_threshold=0.1, concentrated_threshold=0.5, window_size=10)
    assert res["is_shifted"]
    assert res["start_step"] == 390  # last step of uniform (idx 39)
    assert res["end_step"] == 600    # first step of concentrated (idx 60)
    assert "confidence_metrics" in res
    assert "shift_gradient" in res["confidence_metrics"]

def test_visualization_imports_and_mock(tmp_path):
    """Test that visualization functions don't crash when called with empty/mock data."""
    from src.analysis.weight_trajectory import plot_weight_trajectories
    from src.analysis.attention_analysis import plot_attention_heatmap

    # Just verify they run without error on empty dir
    plot_weight_trajectories(tmp_path, tmp_path / "out.png")

    # Test heatmap with mock attention weights
    mock_attn = torch.rand(10, 4, 2, 2)
    plot_attention_heatmap(mock_attn, 0, tmp_path / "heatmap.png")
    assert (tmp_path / "heatmap.png").exists()
