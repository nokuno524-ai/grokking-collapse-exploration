import pytest
import torch
import math
import sys
from pathlib import Path
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))

from src.model import ModularArithmeticTransformer
from src.attention_extraction import (
    extract_attention,
    extract_attention_metrics,
    compute_attention_entropy,
    compute_diag_offdiag_ratio,
    compute_head_specialization
)

@pytest.fixture
def small_model():
    return ModularArithmeticTransformer(
        prime=5,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1
    )

def test_extract_attention_shapes_and_cleanup(small_model):
    batch_size = 4
    seq_len = 2
    inputs = torch.randint(0, 5, (batch_size, seq_len))

    orig_forwards = [layer.self_attn.forward for layer in small_model.transformer.layers]

    with extract_attention(small_model) as extracted:
        assert small_model.transformer.layers[0].self_attn.forward != orig_forwards[0]

        logits = small_model(inputs)

        assert len(extracted) == 1
        assert 0 in extracted
        assert len(extracted[0]) == 1

        weights = extracted[0][0]
        assert weights.shape == (batch_size, 2, seq_len, seq_len)

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))

    assert small_model.transformer.layers[0].self_attn.forward == orig_forwards[0]

def test_attention_entropy():
    uniform_attn = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    entropy = compute_attention_entropy(uniform_attn)
    assert torch.allclose(entropy, torch.tensor([[[math.log(2), math.log(2)]]]))

    onehot_attn = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    entropy = compute_attention_entropy(onehot_attn)
    assert torch.allclose(entropy, torch.zeros_like(entropy), atol=1e-5)

def test_diag_offdiag_ratio():
    diag_attn = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    ratio = compute_diag_offdiag_ratio(diag_attn)
    assert ratio.item() > 1e8

    uniform_attn = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    ratio = compute_diag_offdiag_ratio(uniform_attn)
    assert torch.allclose(ratio, torch.tensor([1.0]), atol=1e-5)

def test_head_specialization():
    h0 = [[1.0, 0.0], [0.0, 1.0]]
    h1 = [[1.0, 0.0], [0.0, 1.0]]
    attn = torch.tensor([[[h0, h1]]])
    spec = compute_head_specialization(attn)
    assert torch.allclose(spec, torch.zeros_like(spec))

    h0 = [[1.0, 0.0], [0.0, 1.0]]
    h1 = [[0.0, 1.0], [1.0, 0.0]]
    attn = torch.tensor([[h0, h1]])
    spec = compute_head_specialization(attn)
    assert torch.allclose(spec, torch.tensor([[0.5, 0.5]]))

def test_determinism_and_metrics_shape(small_model):
    inputs = torch.randint(0, 5, (2, 2))

    torch.manual_seed(42)
    _, metrics1 = extract_attention_metrics(small_model, inputs)

    torch.manual_seed(42)
    _, metrics2 = extract_attention_metrics(small_model, inputs)

    assert torch.allclose(metrics1['weights'], metrics2['weights'])
    assert torch.allclose(metrics1['entropy'], metrics2['entropy'])

    assert metrics1['weights'].shape == (1, 2, 2, 2, 2)
    assert metrics1['entropy'].shape == (1, 2, 2, 2)
    assert metrics1['diag_ratio'].shape == (1, 2, 2)
    assert metrics1['specialization'].shape == (1, 2, 2)