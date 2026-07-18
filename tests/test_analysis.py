import torch
import numpy as np
from analysis.attention_patterns import compute_attention_entropy, identify_circuits

def test_compute_attention_entropy():
    # Shape: (batch_size, n_heads, seq_len, seq_len)
    # Let's create a perfectly sharp attention distribution (entropy 0)
    # and a perfectly uniform attention distribution (max entropy)

    batch_size = 2
    n_heads = 1
    seq_len = 2

    # Sharp: [1.0, 0.0]
    weights_sharp = torch.zeros(batch_size, n_heads, seq_len, seq_len)
    weights_sharp[:, :, :, 0] = 1.0

    # Uniform: [0.5, 0.5]
    weights_uniform = torch.ones(batch_size, n_heads, seq_len, seq_len) * 0.5

    ent_sharp = compute_attention_entropy(weights_sharp)
    ent_uniform = compute_attention_entropy(weights_uniform)

    # Sharp should be ~0
    assert torch.allclose(ent_sharp, torch.zeros_like(ent_sharp), atol=1e-5)

    # Uniform should be ~ -log(0.5) = 0.693
    expected_uniform = -torch.log(torch.tensor(0.5))
    assert torch.allclose(ent_uniform, expected_uniform * torch.ones_like(ent_uniform), atol=1e-5)

def test_identify_circuits():
    batch_size = 1
    n_heads = 2
    seq_len = 2

    # Head 0: Perfect self-attention
    # Head 1: Perfect cross-attention
    weights = torch.zeros(batch_size, n_heads, seq_len, seq_len)

    # Head 0: attends to self
    weights[0, 0, 0, 0] = 1.0
    weights[0, 0, 1, 1] = 1.0

    # Head 1: attends to other
    weights[0, 1, 0, 1] = 1.0
    weights[0, 1, 1, 0] = 1.0

    circuits = identify_circuits(weights)

    # self_attention_score: shape should be (n_heads)
    assert np.allclose(circuits['self_attention_score'], [1.0, 0.0])
    assert np.allclose(circuits['cross_attention_score'], [0.0, 1.0])
