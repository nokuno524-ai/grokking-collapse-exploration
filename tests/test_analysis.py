import pytest
import torch
import numpy as np

# Ensure src modules can be imported
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import ModularArithmeticTransformer
from analysis.attention_analysis import extract_attention_patterns, compute_attention_entropy, track_attention_specialization
from analysis.circuit_analysis import identify_important_circuits, track_circuit_formation_across_collapse
from analysis.weight_analysis import compute_weight_norm_trajectory, compute_singular_value_spectrum, effective_rank_analysis


@pytest.fixture
def mock_model():
    model = ModularArithmeticTransformer(
        prime=59,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=1,
    )
    # Ensure deterministic initialization for tests
    torch.manual_seed(42)
    model._init_weights()
    return model


@pytest.fixture
def mock_data():
    # Batch size 4, seq_len 2
    return torch.randint(0, 59, (4, 2))


@pytest.fixture
def mock_targets(mock_data):
    # targets: (a + b) mod 59
    return (mock_data[:, 0] + mock_data[:, 1]) % 59


def test_extract_attention_patterns(mock_model, mock_data):
    attn_weights = extract_attention_patterns(mock_model, mock_data)

    # Check shape: (batch_size, n_heads, seq_len, seq_len)
    assert attn_weights.shape == (4, 2, 2, 2)

    # Check that weights sum to 1 over the last dimension
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_compute_attention_entropy():
    # Mock attention weights: (batch=1, heads=1, seq=1, seq=2)
    # Uniform distribution: [0.5, 0.5]
    attn_weights = torch.tensor([[[[0.5, 0.5]]]])
    entropy = compute_attention_entropy(attn_weights)

    # Entropy of [0.5, 0.5] is - (0.5*ln(0.5) + 0.5*ln(0.5)) = ln(2) ~ 0.693147
    expected = torch.tensor([[[np.log(2.0)]]], dtype=torch.float32)
    assert torch.allclose(entropy, expected, atol=1e-5)


def test_track_attention_specialization():
    attn_weights_1 = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]]) # batch=1, heads=2, seq=2, seq=2
    # Second checkpoint has a more focused distribution for head 0, uniform for head 1
    attn_weights_2 = torch.tensor([[[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]]])

    avg_entropies = track_attention_specialization([attn_weights_1, attn_weights_2])

    assert len(avg_entropies) == 2
    assert avg_entropies[0].shape == (2,)
    assert avg_entropies[1].shape == (2,)

    # Head 0 in checkpoint 2 should have lower entropy than head 0 in checkpoint 1
    assert avg_entropies[1][0].item() < avg_entropies[0][0].item()
    # Head 1 should have same entropy
    assert np.isclose(avg_entropies[1][1].item(), avg_entropies[0][1].item(), atol=1e-5)


def test_identify_important_circuits(mock_model, mock_data, mock_targets):
    scores = identify_important_circuits(mock_model, mock_data, mock_targets)

    # Should return one score per head
    assert len(scores) == mock_model.n_heads

    # Scores are float values
    assert isinstance(scores[0], float)


def test_track_circuit_formation_across_collapse(mock_model, mock_data, mock_targets):
    models_dict = {
        "pure": [mock_model, mock_model],
        "severe": [mock_model]
    }

    results = track_circuit_formation_across_collapse(models_dict, mock_data, mock_targets)

    assert "pure" in results
    assert "severe" in results
    assert len(results["pure"]) == 2
    assert len(results["severe"]) == 1
    assert len(results["pure"][0]) == mock_model.n_heads


def test_compute_weight_norm_trajectory(mock_model):
    norms = compute_weight_norm_trajectory([mock_model, mock_model])

    assert len(norms) == 2
    assert isinstance(norms[0], float)
    assert np.isclose(norms[0], norms[1])


def test_compute_singular_value_spectrum():
    # Identity matrix has all singular values = 1
    matrix = torch.eye(10)
    s = compute_singular_value_spectrum(matrix)

    assert s.shape == (10,)
    assert torch.allclose(s, torch.ones_like(s))


def test_effective_rank_analysis():
    # Identity matrix has equal singular values, so uniform distribution
    # Normalized S = [0.1, 0.1, ..., 0.1]
    # Entropy = -10 * (0.1 * ln(0.1)) = ln(10)
    # Effective rank = e^ln(10) = 10
    matrix = torch.eye(10)
    er = effective_rank_analysis(matrix)

    assert np.isclose(er, 10.0, atol=1e-5)

    # Rank 1 matrix
    # Singular values: [sqrt(10), 0, ..., 0] -> normalized: [1, 0, ..., 0]
    # Entropy = 0
    # Effective rank = e^0 = 1
    matrix2 = torch.ones(10, 10)
    er2 = effective_rank_analysis(matrix2)

    assert np.isclose(er2, 1.0, atol=1e-5)
