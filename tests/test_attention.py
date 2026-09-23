import pytest
import torch
import torch.nn as nn
import numpy as np

from src.attention_extraction import extract_attention, compute_entropy

def test_extract_attention():
    # Create a small dummy model with MultiheadAttention
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Use same params as our simple model
            self.mha = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
            self.linear = nn.Linear(16, 16)

        def forward(self, x):
            attn_out, _ = self.mha(x, x, x)
            return self.linear(attn_out)

    model = DummyModel()
    model.eval()

    x = torch.randn(4, 5, 16) # batch, seq_len, embed_dim

    with extract_attention(model) as attention_weights:
        out = model(x)

    assert len(attention_weights) == 1
    attn = attention_weights[0]

    # Check shape: (batch_size, num_heads, seq_len, seq_len)
    assert attn.shape == (4, 2, 5, 5)

    # Check it's a valid probability distribution (sums to 1 over last dim)
    sums = attn.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

def test_extract_attention_transformer_layer():
    # Test with actual TransformerEncoderLayer which has fast paths
    model = nn.TransformerEncoderLayer(
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        batch_first=True
    )
    model.eval()

    x = torch.randn(4, 5, 16)

    with extract_attention(model) as attention_weights:
        out = model(x)

    assert len(attention_weights) == 1
    attn = attention_weights[0]

    # Check shape: (batch_size, num_heads, seq_len, seq_len)
    assert attn.shape == (4, 2, 5, 5)

def test_compute_entropy():
    # Create uniform attention
    batch, heads, seq = 2, 3, 4
    attn_uniform = np.ones((batch, heads, seq, seq)) / seq

    entropy_uniform = compute_entropy(attn_uniform)

    # Uniform entropy should be log(seq)
    expected_uniform = np.log(seq)
    np.testing.assert_allclose(entropy_uniform, expected_uniform, rtol=1e-5)

    # Create sharp attention (one token gets probability 1, others 0)
    attn_sharp = np.zeros((batch, heads, seq, seq))
    attn_sharp[..., 0] = 1.0

    entropy_sharp = compute_entropy(attn_sharp)

    # Sharp entropy should be 0
    np.testing.assert_allclose(entropy_sharp, 0.0, atol=1e-5)
