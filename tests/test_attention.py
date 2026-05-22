import torch
import pytest
from src.model import ModularArithmeticTransformer
from src.attention_viz import extract_attention_weights, compute_attention_entropy

def test_extract_attention_weights():
    model = ModularArithmeticTransformer(d_model=32, n_heads=2, n_layers=1)
    x = torch.tensor([[10, 20], [5, 15]])
    attn_list = extract_attention_weights(model, x)

    assert isinstance(attn_list, list)
    assert len(attn_list) == 1

    attn = attn_list[0]
    # Check shape: (batch, n_heads, seq_len, seq_len)
    assert attn.shape == (2, 2, 2, 2)

    # Check if probabilities sum to 1 over last dim
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))

def test_compute_attention_entropy():
    # Construct a deterministic attention distribution
    # 2 batches, 1 head, 2x2 sequence
    attn = torch.tensor([[[
        [1.0, 0.0],
        [0.5, 0.5]
    ]]])

    entropy = compute_attention_entropy(attn)
    assert entropy.shape == (1, 1, 2)

    # Entropy of [1.0, 0.0] should be ~0
    assert entropy[0, 0, 0].item() < 1e-5

    # Entropy of [0.5, 0.5] should be -log(0.5) ~ 0.693
    import math
    expected_entropy = -math.log(0.5)
    assert abs(entropy[0, 0, 1].item() - expected_entropy) < 1e-5
