import os
import torch
import numpy as np
import pytest
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ModularArithmeticTransformer
from visualization.attention_evolution import compute_attention_patterns, load_checkpoint
from visualization.weight_analysis import compute_weight_norm, extract_singular_values
from analysis.circuit_detection import compute_effective_rank, compute_participation_ratio

def create_mock_checkpoint(filepath: str):
    """Create a mock checkpoint file for testing."""
    model = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2, d_ff=32)
    ckpt = {
        'step': 100,
        'model_state': model.state_dict(),
        'config': {
            'prime': 7,
            'd_model': 16,
            'n_heads': 2,
            'd_ff': 32,
            'n_layers': 1
        }
    }
    torch.save(ckpt, filepath)
    return model

@pytest.fixture
def mock_ckpt_dir(tmp_path):
    ckpt_path = os.path.join(tmp_path, "checkpoint_100.pt")
    create_mock_checkpoint(ckpt_path)
    return tmp_path

def test_compute_attention_patterns(mock_ckpt_dir):
    ckpt_path = os.path.join(mock_ckpt_dir, "checkpoint_100.pt")
    model, config, step = load_checkpoint(ckpt_path)

    assert step == 100
    assert config['prime'] == 7

    attn_grid = compute_attention_patterns(model, prime=7)

    # Check shape: (n_heads, prime, prime)
    assert attn_grid.shape == (2, 7, 7)

    # Check probabilities sum to 1 along dim -1, or at least are in [0, 1]
    assert np.all(attn_grid >= 0) and np.all(attn_grid <= 1)

def test_weight_analysis_metrics(mock_ckpt_dir):
    ckpt_path = os.path.join(mock_ckpt_dir, "checkpoint_100.pt")
    ckpt = torch.load(ckpt_path, weights_only=True)
    weights = ckpt['model_state']

    # Test weight norm
    norm = compute_weight_norm(weights)
    assert norm > 0
    assert isinstance(norm, float)

    # Test singular values
    s = extract_singular_values(weights, 'transformer.layers.0.self_attn.in_proj_weight')
    assert len(s) > 0
    assert isinstance(s, np.ndarray)
    assert np.all(s >= 0) # singular values should be non-negative

def test_circuit_metrics():
    # Create a random matrix
    w = torch.randn(10, 10)

    eff_rank = compute_effective_rank(w)
    pr = compute_participation_ratio(w)

    assert eff_rank > 0
    assert pr > 0

    # Test flat matrix
    w_flat = torch.zeros(10, 10)
    w_flat[0, 0] = 1.0

    eff_rank_flat = compute_effective_rank(w_flat)
    pr_flat = compute_participation_ratio(w_flat)

    # Effective rank of rank-1 matrix is 1.0
    assert np.isclose(eff_rank_flat, 1.0, atol=1e-5)
    assert np.isclose(pr_flat, 1.0, atol=1e-5)
