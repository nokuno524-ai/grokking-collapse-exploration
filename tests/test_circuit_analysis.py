"""Tests for circuit-level mechanistic analysis tools."""

import torch
import torch.nn.functional as F
import numpy as np
from src.model import ModularArithmeticTransformer
from src.analysis.circuit_analysis import (
    extract_qkv_weights,
    compute_manual_attention,
    manual_transformer_forward,
    CircuitDiscoveryTool,
    WeightDecomposition,
)

def test_extract_qkv_weights():
    d_model = 12
    n_heads = 4

    in_proj_weight = torch.randn(3 * d_model, d_model)
    in_proj_bias = torch.randn(3 * d_model)

    w_q, w_k, w_v, b_q, b_k, b_v = extract_qkv_weights(
        in_proj_weight, in_proj_bias, d_model, n_heads
    )

    assert w_q.shape == (d_model, d_model)
    assert w_k.shape == (d_model, d_model)
    assert w_v.shape == (d_model, d_model)
    assert b_q.shape == (d_model,)

    assert torch.allclose(w_q, in_proj_weight[:d_model, :])
    assert torch.allclose(w_k, in_proj_weight[d_model:2*d_model, :])
    assert torch.allclose(w_v, in_proj_weight[2*d_model:, :])


def test_manual_transformer_forward_equivalence():
    """Test that manual forward pass exactly matches the native PyTorch model."""
    torch.manual_seed(42)
    model = ModularArithmeticTransformer(
        prime=11, d_model=32, n_heads=4, d_ff=64, n_layers=1, dropout=0.0
    )
    model.eval()

    # (batch, seq_len)
    x = torch.randint(0, 11, (5, 2))

    # Native forward
    with torch.no_grad():
        native_logits = model(x)

    # Manual forward
    with torch.no_grad():
        manual_logits, attn_probs = manual_transformer_forward(model, x)

    assert torch.allclose(native_logits, manual_logits, atol=1e-5), "Manual and native logits differ"
    assert len(attn_probs) == 1
    assert attn_probs[0].shape == (5, 4, 2, 2)  # (batch, n_heads, seq_len, seq_len)
    # Check that attention probabilities sum to 1 over last dim
    assert torch.allclose(attn_probs[0].sum(dim=-1), torch.ones(5, 4, 2))


def test_circuit_discovery_tool():
    """Test the CircuitDiscoveryTool with ablations."""
    torch.manual_seed(42)
    model = ModularArithmeticTransformer(
        prime=11, d_model=32, n_heads=4, d_ff=64, n_layers=1, dropout=0.0
    )

    x = torch.randint(0, 11, (5, 2))
    y = torch.randint(0, 11, (5,))

    tool = CircuitDiscoveryTool(model)
    importance = tool.compute_head_importance(x, y)

    assert importance.shape == (1, 4)
    # Importance scores should not be all exactly zero (unless model is trivial)
    assert not np.allclose(importance, 0.0)


def test_weight_decomposition():
    """Test SVD decomposition and space comparison."""
    torch.manual_seed(42)
    W = torch.randn(64, 64)

    # Get components
    U, S, Vh = WeightDecomposition.get_svd_components(W, k=10)
    assert U.shape == (64, 10)
    assert S.shape == (10,)
    assert Vh.shape == (10, 64)

    # Ensure S is ordered descending
    assert torch.all(S[:-1] >= S[1:])

    # Space overlap with itself should be 1.0
    overlap = WeightDecomposition.compare_singular_spaces(U, U)
    assert abs(overlap - 1.0) < 1e-5

    # Space overlap with orthogonal space should be 0.0
    # Create U2 orthogonal to U
    U_full, _, _ = torch.linalg.svd(W, full_matrices=True)
    U2 = U_full[:, 10:20]  # orthogonal to first 10
    overlap_orth = WeightDecomposition.compare_singular_spaces(U, U2)
    assert abs(overlap_orth) < 1e-5
