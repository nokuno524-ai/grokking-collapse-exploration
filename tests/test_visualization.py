import torch
import numpy as np
import pytest
from visualization.attention_evolution import compute_attention_entropy, compute_attention_similarity
from visualization.training_dynamics import detect_grokking_transition

def test_compute_attention_entropy():
    # Test perfectly uniform distribution (max entropy)
    n = 4
    attn_uniform = torch.ones((1, 1, n)) / n
    # Each item is 1/n, so p * log(p) is (1/n)*log(1/n). Sum over n items is n * (1/n) * log(1/n) = log(1/n). Entropy = -log(1/n)
    expected_entropy = -torch.log(torch.tensor(1/n))
    entropy = compute_attention_entropy(attn_uniform)
    assert torch.allclose(entropy, expected_entropy)

    # Test completely peaked distribution (zero entropy)
    attn_peaked = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    entropy = compute_attention_entropy(attn_peaked)
    assert torch.allclose(entropy, torch.tensor(0.0), atol=1e-5)

def test_compute_attention_similarity():
    # Test identical attention matrices (similarity = 1)
    attn1 = torch.rand((2, 4, 10, 10))
    attn2 = attn1.clone()
    sim = compute_attention_similarity(attn1, attn2)
    assert np.isclose(sim, 1.0)

    # Test orthogonal attention matrices (similarity = 0)
    # Using 1D for simplicity
    a1 = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    a2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    sim = compute_attention_similarity(a1, a2)
    assert np.isclose(sim, 0.0)

def test_detect_grokking_transition():
    # Test standard grokking curve
    # 10 steps of random (0.1), then grokking to perfect (1.0)
    accs = [0.1]*10 + [0.5, 0.8, 1.0, 1.0]
    transition = detect_grokking_transition(accs, random_threshold=0.15, perfect_threshold=0.95)
    assert transition is not None
    assert transition[0] == 9   # Last step where acc < 0.15
    assert transition[1] == 12  # First step where acc > 0.95

    # Test no grokking
    accs_no_grok = [0.1]*20
    assert detect_grokking_transition(accs_no_grok) is None

    # Test already grokked from step 0 (edge case)
    accs_perfect = [1.0]*20
    transition = detect_grokking_transition(accs_perfect)
    assert transition is not None
    assert transition[0] == 0
    assert transition[1] == 0
