import pytest
import numpy as np
import torch
import math
import os
from pathlib import Path

from analysis.grokking_detector import extract_phases
from analysis.weight_space import compute_effective_rank, compute_cosine_distance, compute_activation_cka
from analysis.attention_evolution import compute_attention_entropy

def test_grokking_detector():
    """Test the grokking detector with a synthetic accuracy S-curve."""
    steps = np.arange(0, 1000, 10)
    # Sigmoid to simulate grokking
    # S-curve centered at step 500
    acc = 1.0 / (1.0 + np.exp(-(steps - 500) / 20.0))
    # Before 500, it's low. After 500, it's high.

    result = extract_phases(acc, steps, window_length=11)
    assert result['grokking_step'] is not None
    # Maximum acceleration is slightly before the inflection point
    assert 400 <= result['grokking_step'] <= 500
    assert result['grokking_gap'] > 0.4

def test_grokking_detector_no_grokking():
    """Test the grokking detector on a flat curve."""
    steps = np.arange(0, 1000, 10)
    acc = np.full_like(steps, 0.1, dtype=float)

    result = extract_phases(acc, steps, window_length=11)
    assert result['grokking_step'] is None

def test_effective_rank():
    """Test effective rank computation using Shannon entropy."""
    # Singular values: [1, 0, 0] -> p=[1, 0, 0] -> entropy=0 -> exp(0)=1
    s1 = torch.tensor([1.0, 0.0, 0.0])
    rank1 = compute_effective_rank(s1)
    assert math.isclose(rank1, 1.0, rel_tol=1e-4)

    # Singular values: [1, 1, 1] -> p=[1/3, 1/3, 1/3] -> entropy=ln(3) -> exp(ln(3))=3
    s2 = torch.tensor([1.0, 1.0, 1.0])
    rank2 = compute_effective_rank(s2)
    assert math.isclose(rank2, 3.0, rel_tol=1e-4)

def test_cosine_distance():
    """Test global cosine distance between state dicts."""
    state1 = {'w1': torch.tensor([1.0, 0.0]), 'w2': torch.tensor([0.0, 1.0])}
    state2 = {'w1': torch.tensor([1.0, 0.0]), 'w2': torch.tensor([0.0, 1.0])}
    # Same state -> distance 0
    dist = compute_cosine_distance(state1, state2)
    assert dist < 1e-5

    state3 = {'w1': torch.tensor([-1.0, 0.0]), 'w2': torch.tensor([0.0, -1.0])}
    # Opposite state -> similarity -1 -> distance 2
    dist2 = compute_cosine_distance(state1, state3)
    assert math.isclose(dist2, 2.0, rel_tol=1e-4)

def test_activation_cka():
    """Test CKA computation on activations."""
    # Random orthogonal matrices -> CKA ~ 0 for uncorrelated
    np.random.seed(42)
    acts1 = torch.randn(100, 10)
    acts2 = torch.randn(100, 10)
    cka_diff = compute_activation_cka(acts1, acts2)
    assert 0.0 <= cka_diff <= 1.0

    # Same matrix -> CKA = 1
    cka_same = compute_activation_cka(acts1, acts1)
    assert math.isclose(cka_same, 1.0, rel_tol=1e-4)

def test_attention_entropy():
    """Test attention entropy computation."""
    # (batch, n_heads, seq, seq)
    # Perfect uniform attention -> p = 1/seq -> entropy = ln(seq)
    batch, heads, seq = 2, 3, 4
    attn_uniform = torch.ones(batch, heads, seq, seq) / seq

    entropy = compute_attention_entropy(attn_uniform)
    assert entropy.shape == (heads,)
    # Should be close to ln(4)
    expected = math.log(4)
    assert torch.allclose(entropy, torch.full_like(entropy, expected), atol=1e-4)

    # Perfect focused attention -> p = 1 for one token, 0 for rest -> entropy = 0
    attn_focus = torch.zeros(batch, heads, seq, seq)
    attn_focus[:, :, :, 0] = 1.0
    entropy2 = compute_attention_entropy(attn_focus)
    assert torch.allclose(entropy2, torch.zeros_like(entropy2), atol=1e-4)
