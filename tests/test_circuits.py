import torch
import pytest
from src.model import ModularArithmeticTransformer
from analysis.circuits import extract_attention_patterns, compare_circuit_structures, identify_grokking_circuits

def test_extract_attention_patterns():
    model = ModularArithmeticTransformer()
    x = torch.randint(0, 59, (4, 2))

    patterns = extract_attention_patterns(model, x)
    # expected shape: (batch_size, n_heads, seq_len, seq_len)
    # wait, MultiheadAttention without average_attn_weights=False usually returns (batch, seq_len, seq_len)
    # let's check shape logic from PyTorch doc:
    # if average_attn_weights=False, returns (batch, num_heads, seq_len, seq_len)
    assert patterns.shape == (4, model.n_heads, 2, 2)
    # Check that they are valid probabilities
    assert torch.allclose(patterns.sum(dim=-1), torch.ones_like(patterns.sum(dim=-1)))

def test_compare_circuit_structures():
    patterns1 = torch.rand(4, 4, 2, 2)
    patterns2 = torch.rand(4, 4, 2, 2)

    metrics = compare_circuit_structures(patterns1, patterns2)
    assert "l2_diff" in metrics
    assert "cosine_similarity" in metrics

    metrics_same = compare_circuit_structures(patterns1, patterns1)
    assert metrics_same["l2_diff"] == 0.0
    assert abs(metrics_same["cosine_similarity"] - 1.0) < 1e-5

def test_identify_grokking_circuits():
    # 4 heads, seq_len 2
    pre = torch.zeros(2, 4, 2, 2)
    post = torch.zeros(2, 4, 2, 2)

    # head 0 changes a lot
    post[:, 0] = 1.0

    # head 1 changes a little
    post[:, 1] = 0.1

    specialized = identify_grokking_circuits(pre, post, threshold=0.5)

    assert 0 in specialized
    assert 1 not in specialized
