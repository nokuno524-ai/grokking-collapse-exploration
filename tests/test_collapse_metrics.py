import torch
import pytest
import math
from src.collapse_metrics import (
    representation_collapse_score,
    gradient_collapse_score,
    output_diversity_index,
    weight_matrix_conditioning,
    attention_pattern_collapse
)

def test_representation_collapse_score():
    # Full rank random matrix should have high score
    torch.manual_seed(42)
    full_rank = torch.randn(100, 50)
    score_full = representation_collapse_score(full_rank)

    # Low rank matrix (rank 1) should have low score
    low_rank = torch.ones(100, 1) @ torch.ones(1, 50)
    score_low = representation_collapse_score(low_rank)

    assert score_full > score_low
    assert score_low < 2.0  # Rank 1 should have effective rank close to 1

    # Test 3D input
    input_3d = torch.randn(10, 20, 30)
    score_3d = representation_collapse_score(input_3d)
    assert score_3d > 1.0


def test_gradient_collapse_score():
    # Orthogonal gradients should have ~0 similarity
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([0.0, 1.0])
    score_ortho = gradient_collapse_score([g1, g2])
    assert abs(score_ortho) < 1e-6

    # Identical gradients should have ~1 similarity
    g3 = torch.tensor([1.0, 1.0])
    g4 = torch.tensor([2.0, 2.0])  # Scaled version is still identical direction
    score_identical = gradient_collapse_score([g3, g4])
    assert abs(score_identical - 1.0) < 1e-6

    # Single gradient should return 0
    assert gradient_collapse_score([g1]) == 0.0


def test_output_diversity_index():
    # Uniform distribution should have max entropy
    # log(num_classes) = log(10) ≈ 2.3
    logits_uniform = torch.zeros(100, 10)
    entropy_uniform = output_diversity_index(logits_uniform)
    assert abs(entropy_uniform - math.log(10)) < 1e-4

    # Collapsed distribution (predicts class 0 always) should have ~0 entropy
    logits_collapsed = torch.zeros(100, 10)
    logits_collapsed[:, 0] = 100.0  # Huge logit for class 0
    entropy_collapsed = output_diversity_index(logits_collapsed)
    assert entropy_collapsed < 1e-4

    assert entropy_uniform > entropy_collapsed


def test_weight_matrix_conditioning():
    # Identity matrix should have condition number 1
    w_identity = torch.eye(10)
    cond_identity = weight_matrix_conditioning(w_identity)
    assert abs(cond_identity - 1.0) < 1e-5

    # Matrix with wildly different singular values should have high condition number
    w_ill = torch.diag(torch.tensor([1000.0, 1.0, 0.001]))
    cond_ill = weight_matrix_conditioning(w_ill)
    assert cond_ill > 1e5

    # Test non-square matrices
    w_nonsquare = torch.randn(10, 5)
    cond_nonsquare = weight_matrix_conditioning(w_nonsquare)
    assert cond_nonsquare >= 1.0


def test_attention_pattern_collapse():
    # Diverse attention patterns across batch
    batch_size = 10
    seq_len = 5
    num_heads = 2

    # Random attention (simulate diverse patterns)
    attn_diverse = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
    # List of individual batch items
    attn_diverse_list = [attn_diverse[i] for i in range(batch_size)]
    score_diverse = attention_pattern_collapse(attn_diverse_list)

    # Identical attention patterns
    attn_base = torch.softmax(torch.randn(num_heads, seq_len, seq_len), dim=-1)
    attn_identical_list = [attn_base.clone() for _ in range(batch_size)]
    score_identical = attention_pattern_collapse(attn_identical_list)

    # Score is 1.0 for identical patterns (complete collapse)
    assert abs(score_identical - 1.0) < 1e-5
    # Score for diverse patterns should be < 1.0
    assert score_diverse < score_identical
