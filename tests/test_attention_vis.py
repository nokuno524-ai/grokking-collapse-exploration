import pytest
import torch
from src.analysis.attention_vis import compute_attention_entropy, compute_head_specialization

def test_compute_attention_entropy():
    # shape: (batch, heads, seq, seq)
    # perfectly uniform attention over 2 tokens should have entropy -ln(0.5) ~ 0.693
    uniform_attn = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    entropy = compute_attention_entropy(uniform_attn)

    assert entropy.shape == (1, 1, 2)
    assert torch.allclose(entropy, torch.tensor([[[0.6931, 0.6931]]]), atol=1e-4)

    # perfectly peaked attention
    peaked_attn = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    entropy2 = compute_attention_entropy(peaked_attn)
    assert torch.allclose(entropy2, torch.tensor([[[0.0, 0.0]]]), atol=1e-4)

def test_compute_head_specialization():
    # shape: (batch, heads, seq, seq)
    # Head perfectly attends to pos 0 for both queries
    attn1 = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    spec1 = compute_head_specialization(attn1, None)
    assert spec1.shape == (1, 1)
    assert torch.allclose(spec1, torch.tensor([[1.0]]))

    # Uniform attention
    attn2 = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    spec2 = compute_head_specialization(attn2, None)
    assert torch.allclose(spec2, torch.tensor([[0.0]]))


def test_compute_attention_diff():
    from src.analysis.attention_vis import compute_attention_diff
    pre = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    post = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    diff = compute_attention_diff(pre, post)

    assert diff.shape == (1, 1)
    # diff matrix: [[[0.5, 0.5], [0.5, 0.5]]] -> mean is 0.5
    assert torch.allclose(diff, torch.tensor([[0.5]]))
