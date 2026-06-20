import pytest
import torch
import numpy as np

# Add src to path explicitly to avoid import issues
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import ModularArithmeticTransformer
from analysis.induction_heads import compute_induction_score, extract_attention_weights
from analysis.circuit_complexity import compute_circuit_density, extract_activations
from analysis.neuron_analysis import compute_polysemanticity, get_ffn_activations
from analysis.info_flow import compute_mi_proxy, get_representations


@pytest.fixture
def model():
    # Small model for fast testing
    return ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)


@pytest.fixture
def batch_inputs():
    return torch.tensor([[10, 20], [30, 40], [5, 15]])


@pytest.fixture
def batch_targets():
    return torch.tensor([30, 11, 20])


def test_induction_score(model, batch_inputs):
    attn_weights = extract_attention_weights(model, batch_inputs)
    assert attn_weights.shape == (3, 2, 2, 2)  # (batch, n_heads, seq_len, seq_len)

    score = compute_induction_score(attn_weights)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_circuit_density(model, batch_inputs):
    acts = extract_activations(model, batch_inputs)
    assert 'v' in acts
    assert acts['v'].shape == (3, 2, 2, 16)  # (batch, seq, n_heads, d_head)

    density = compute_circuit_density(model, acts)
    assert isinstance(density, float)
    assert density >= 0.0


def test_neuron_polysemanticity(model, batch_inputs):
    acts = get_ffn_activations(model, batch_inputs)
    assert acts.shape == (3, 2, 64)  # (batch, seq, d_ff)

    score = compute_polysemanticity(acts)
    assert isinstance(score, float)
    assert score > 0.0


def test_information_flow(model, batch_inputs, batch_targets):
    reps = get_representations(model, batch_inputs)
    assert 'embedding' in reps
    assert 'post_attn' in reps
    assert 'post_ffn' in reps

    assert reps['embedding'].shape == (3, 32)

    score = compute_mi_proxy(reps['embedding'], batch_targets)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
