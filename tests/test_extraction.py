import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer
from src.attention.extraction import extract_attention, compute_entropy

def test_extract_attention():
    torch.manual_seed(42)
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=4, n_layers=2)

    # Input batch of size 3, sequence length 2
    x = torch.randint(0, 59, (3, 2))

    with extract_attention(model) as attn_weights:
        model(x)

    assert len(attn_weights) == 2  # 2 layers
    for weight in attn_weights:
        # Check shape: (batch, n_heads, seq_len, seq_len)
        # Note: Depending on batch_first, MultiheadAttention might return (batch, num_heads, L, S)
        # or (batch, L, S) if not averaged.
        assert weight.ndim == 4
        assert weight.shape[0] == 3
        assert weight.shape[1] == 4
        assert weight.shape[2] == 2
        assert weight.shape[3] == 2

        # Check probabilities sum to 1
        sums = weight.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))

def test_compute_entropy():
    # Batch=1, Heads=1, Seq=2, Seq=2
    # Uniform attention
    w1 = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    ent1 = compute_entropy(w1)
    # Expected entropy: - (0.5*ln(0.5) + 0.5*ln(0.5)) = ln(2)
    expected1 = torch.tensor([[[-0.5 * torch.log(torch.tensor(0.5)) * 2,
                                -0.5 * torch.log(torch.tensor(0.5)) * 2]]])
    assert torch.allclose(ent1, expected1)

    # Sharp attention
    w2 = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    ent2 = compute_entropy(w2)
    # Expected entropy should be 0
    assert torch.allclose(ent2, torch.zeros_like(ent2), atol=1e-7)

def test_compute_entropy_with_mask():
    # Batch=1, Heads=1, Seq=3, Seq=3
    # Padding on last token
    w = torch.tensor([[[
        [0.3, 0.3, 0.4],
        [0.4, 0.4, 0.2],
        [0.1, 0.1, 0.8]  # This is the padding token's query
    ]]])
    mask = torch.tensor([[True, True, False]])

    ent = compute_entropy(w, mask)

    # The padding token query (idx 2) should have exactly 0 entropy
    assert ent[0, 0, 2] == 0.0

    # For token 0, it should renormalize [0.3, 0.3] -> [0.5, 0.5]
    expected_0 = - (0.5 * torch.log(torch.tensor(0.5)) * 2)
    assert torch.allclose(ent[0, 0, 0], expected_0, atol=1e-5)
