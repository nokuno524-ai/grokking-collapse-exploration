import pytest
import torch
import sys
import os
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ModularArithmeticTransformer
from analysis.attention_pattern_analysis import extract_attention_weights, compute_attention_entropy, measure_attention_similarity
from analysis.weight_analysis import analyze_weight_rank

def test_extract_attention_weights():
    """Test manual attention weight extraction."""
    model = ModularArithmeticTransformer(d_model=32, n_heads=4, d_ff=64)
    model.eval()

    batch_size = 2
    x = torch.randint(0, 59, (batch_size, 2))

    attn = extract_attention_weights(model, x)

    # Check shape: (batch, n_heads, seq_len, seq_len)
    assert attn.shape == (batch_size, 4, 2, 2)

    # Check probabilities sum to 1 over last dim
    sums = attn.sum(dim=-1)
    assert bool(torch.allclose(sums, torch.ones_like(sums))) is True

    # Check all weights >= 0
    assert bool(torch.all(attn >= 0)) is True

def test_compute_attention_entropy():
    """Test entropy computation handles uniform and concentrated distributions."""
    # Dummy weights: (batch, n_heads, seq_len, seq_len)
    batch_size = 1
    n_heads = 2
    seq_len = 2

    # Head 0: Uniform attention [0.5, 0.5]
    # Head 1: Concentrated attention [1.0, 0.0]
    weights = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]], [[1.0, 0.0], [1.0, 0.0]]]])

    entropy = compute_attention_entropy(weights)

    # Shape should be (batch, n_heads, seq_len)
    assert entropy.shape == (1, 2, 2)

    # Head 0 (uniform) should have entropy ~ -log(0.5) ~ 0.693
    assert np.isclose(entropy[0, 0, 0].item(), 0.693147)

    # Head 1 (concentrated) should have entropy ~ 0
    assert np.isclose(entropy[0, 1, 0].item(), 0.0, atol=1e-5)

def test_measure_attention_similarity():
    """Test attention similarity computes MSE between averaged attentions."""
    model1 = ModularArithmeticTransformer(d_model=32, n_heads=2)
    model2 = ModularArithmeticTransformer(d_model=32, n_heads=2)

    x = torch.randint(0, 59, (2, 2))

    similarity = measure_attention_similarity(model1, model2, x)
    assert isinstance(similarity, float)
    assert similarity >= 0.0

    # Model against itself should have similarity 0
    self_similarity = measure_attention_similarity(model1, model1, x)
    assert np.isclose(self_similarity, 0.0)

def test_analyze_weight_rank():
    """Test weight rank analysis computes effective rank (entropy of SVD)."""
    model = ModularArithmeticTransformer(d_model=32, n_heads=2)

    ranks = analyze_weight_rank(model)

    assert 'token_embed' in ranks
    assert 'output_head' in ranks
    assert 'attn_q' in ranks

    for k, v in ranks.items():
        assert isinstance(v, float)
        assert v > 0.0
