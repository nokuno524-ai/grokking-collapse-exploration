import torch
import pytest
from src.model import ModularArithmeticTransformer
from src.attention_viz import (
    get_attention_patterns,
    track_head_specialization,
    attention_head_diversity,
    ablate_attention_head,
    measure_head_importance
)

@pytest.fixture
def mock_model():
    # Use small model for fast testing
    return ModularArithmeticTransformer(
        prime=5,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1
    )

def test_get_attention_patterns(mock_model):
    inputs = torch.tensor([[1, 2], [3, 4]])

    with torch.no_grad():
        attn = get_attention_patterns(mock_model, inputs)

    # Shape: (batch, n_heads, seq_len, seq_len)
    assert attn.shape == (2, 2, 2, 2)

    # Attention probabilities should sum to 1 over key dimension
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))

def test_track_head_specialization(mock_model):
    inputs = torch.tensor([[1, 2], [3, 4]])
    targets = torch.tensor([3, 2])

    with torch.no_grad():
        specs = track_head_specialization(mock_model, inputs, targets)

    assert "pos_0" in specs
    assert "pos_1" in specs
    assert len(specs["pos_0"]) == 2  # n_heads
    assert len(specs["pos_1"]) == 2

    # Probabilities should roughly sum to 1
    total_prob = specs["pos_0"] + specs["pos_1"]
    assert torch.allclose(torch.tensor(total_prob), torch.ones(2))

def test_attention_head_diversity(mock_model):
    inputs = torch.tensor([[1, 2], [3, 4]])

    with torch.no_grad():
        div = attention_head_diversity(mock_model, inputs)

    assert isinstance(div, float)
    assert -1.0 - 1e-5 <= div <= 1.0 + 1e-5

def test_ablate_attention_head(mock_model):
    # Ablate head 0
    ablated = ablate_attention_head(mock_model, head_idx=0)

    # Check that out_proj weights for head 0 are zero
    layer = ablated.transformer.layers[0]
    out_proj_weight = layer.self_attn.out_proj.weight

    # d_model=16, n_heads=2 -> head_dim=8
    # Head 0 columns (0:8) should be 0
    assert torch.allclose(out_proj_weight[:, 0:8], torch.zeros_like(out_proj_weight[:, 0:8]))

    # Head 1 columns (8:16) should be non-zero (same as original)
    orig_layer = mock_model.transformer.layers[0]
    orig_out_weight = orig_layer.self_attn.out_proj.weight
    assert torch.allclose(out_proj_weight[:, 8:16], orig_out_weight[:, 8:16])
