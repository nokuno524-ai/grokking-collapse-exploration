import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import json

from src.model import ModularArithmeticTransformer
from src.attention_extraction import (
    extract_attention,
    compute_entropy,
    compute_diagonal_dominance,
    run_attention_extraction,
    cache_attention_extraction
)

def test_extract_attention_hook_registration():
    model = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1)

    # Run normally
    x = torch.randint(0, 7, (4, 2))
    _ = model(x)

    # Run with context manager
    with extract_attention(model) as extracted:
        _ = model(x)

    # Check that it extracted
    assert len(extracted) == 1

    module_name = list(extracted.keys())[0]
    assert "transformer" in module_name or "self_attn" in module_name

    attn_weights = extracted[module_name][0]
    assert isinstance(attn_weights, np.ndarray)

    # Shape: (batch, num_heads, seq_len, seq_len)
    assert attn_weights.shape == (4, 2, 2, 2)

    # Check probabilities sum to 1
    sums = np.sum(attn_weights, axis=-1)
    np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5)

    # Check original forward is restored
    prev_fastpath = torch.backends.mha.get_fastpath_enabled()
    assert prev_fastpath == torch.backends.mha.get_fastpath_enabled()


def test_compute_entropy():
    # Shape: (batch=1, heads=2, seq=2, seq=2)
    # Head 0: Uniform attention (max entropy)
    # Head 1: One-hot attention (min entropy)
    attn = np.array([[[
        [0.5, 0.5],
        [0.5, 0.5]
    ], [
        [1.0, 0.0],
        [0.0, 1.0]
    ]]])

    entropy = compute_entropy(attn)
    assert entropy.shape == (1, 2, 2)

    # Head 0: -0.5*log2(0.5)*2 = 1.0
    np.testing.assert_allclose(entropy[0, 0], [1.0, 1.0], rtol=1e-5)

    # Head 1: 0 (or close to it due to eps)
    np.testing.assert_allclose(entropy[0, 1], [0.0, 0.0], atol=1e-5)

def test_compute_diagonal_dominance():
    attn = np.array([[[
        [0.8, 0.2],
        [0.1, 0.9]
    ], [
        [0.1, 0.9],
        [0.8, 0.2]
    ]]])

    dom = compute_diagonal_dominance(attn)
    assert dom.shape == (1, 2)

    # Head 0 trace = 1.7 / 2 = 0.85
    np.testing.assert_allclose(dom[0, 0], 0.85, rtol=1e-5)

    # Head 1 trace = 0.3 / 2 = 0.15
    np.testing.assert_allclose(dom[0, 1], 0.15, rtol=1e-5)

def test_cache_attention_extraction_error_paths(tmp_path):
    # Test with non-existent run directory
    with pytest.raises(FileNotFoundError):
        run_attention_extraction(tmp_path / "fake", step=100)

    # Setup mock run directory
    run_dir = tmp_path / "mock_run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        run_attention_extraction(run_dir, step=None)

    # Save a fake checkpoint to the directory
    model = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1)
    torch.save({"model_state": model.state_dict(), "config": {"prime": 7, "d_model": 16, "n_heads": 2, "d_ff": 32, "n_layers": 1}}, run_dir / "checkpoint_10.pt")

    with pytest.raises(FileNotFoundError):
        run_attention_extraction(run_dir, step=5)  # Missing step

    # Should work for existing step
    res = run_attention_extraction(run_dir, step=10, batch_size=4)
    assert res["step"] == 10
    assert res["prime"] == 7
    assert len(res["batch_inputs"]) == 4

    # Test cache extraction
    out_path = tmp_path / "cache.npz"
    cache_attention_extraction(run_dir, out_path, step=10, batch_size=4)
    assert out_path.exists()

    cached = np.load(out_path)
    assert cached["step"] == 10
    assert "attn_transformer.layers.0.self_attn" in cached
