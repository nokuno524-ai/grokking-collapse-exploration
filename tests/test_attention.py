import pytest
import torch
import numpy as np
from pathlib import Path

from src.model import ModularArithmeticTransformer
from src.analysis.attention import (
    AttentionExtractor,
    compute_attention_entropy,
    compute_head_specialization_clustering
)
from src.viz.attention import (
    plot_attention_entropy_over_time,
    plot_head_specialization_heatmap,
    plot_diagnostic_token_traces
)

@pytest.fixture
def tiny_model():
    # Very small model for fast testing
    return ModularArithmeticTransformer(
        prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=2
    )

@pytest.fixture
def probe_data():
    return torch.randint(0, 5, (4, 2))

def test_attention_extractor(tiny_model, probe_data):
    tiny_model.eval()

    with AttentionExtractor(tiny_model) as extractor:
        _ = tiny_model(probe_data)

    assert len(extractor.maps) == 2, "Should extract 2 layers"

    for i in range(2):
        attn = extractor.maps[i]
        # shape: (batch, n_heads, seq, seq)
        assert attn.shape == (4, 2, 2, 2)

        # Check row normalized (sums to 1 along last dim)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

        # Check detached
        assert not attn.requires_grad

def test_attention_entropy():
    # Create deterministic attention matrix (batch=1, head=1, seq=2, seq=2)
    # [ [0.5, 0.5], [1.0, 0.0] ]
    attn = torch.tensor([[[[0.5, 0.5], [1.0, 0.0]]]])
    entropy = compute_attention_entropy(attn)

    assert entropy.shape == (1, 1, 2)

    # 0.5, 0.5 -> ~0.693
    # 1.0, 0.0 -> ~0.0
    assert torch.isclose(entropy[0, 0, 0], torch.tensor(0.6931), atol=1e-3)
    assert torch.isclose(entropy[0, 0, 1], torch.tensor(0.0), atol=1e-3)

def test_head_specialization_clustering():
    # 2 layers, 2 heads each -> 4 heads total
    # head 0: all attend to 0
    # head 1: all attend to 1
    # head 2: all attend to 0
    # head 3: all attend to 1
    map1 = torch.zeros(1, 2, 2, 2)
    map1[:, 0, :, 0] = 1.0
    map1[:, 1, :, 1] = 1.0

    map2 = torch.zeros(1, 2, 2, 2)
    map2[:, 0, :, 0] = 1.0
    map2[:, 1, :, 1] = 1.0

    labels = compute_head_specialization_clustering([map1, map2], n_clusters=2)

    assert len(labels) == 4
    # heads with same patterns should cluster together
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]

def test_visualization_functions(tmp_path):
    # Entropy over time
    steps = [0, 10, 20]
    entropies = [1.0, 0.8, 0.2]
    png1 = tmp_path / "entropy.png"
    csv1 = tmp_path / "entropy.csv"
    plot_attention_entropy_over_time(steps, entropies, "Test Entropy", png1, csv1)

    assert png1.exists()
    assert csv1.exists()

    # Heatmap
    labels = np.array([0, 1, 1, 0])
    png2 = tmp_path / "heatmap.png"
    csv2 = tmp_path / "heatmap.csv"
    plot_head_specialization_heatmap(labels, 2, 2, "Test Heatmap", png2, csv2)

    assert png2.exists()
    assert csv2.exists()

    # Traces
    traces = {"trace1": [0.1, 0.2, 0.3], "trace2": [0.9, 0.8, 0.7]}
    png3 = tmp_path / "traces.png"
    csv3 = tmp_path / "traces.csv"
    plot_diagnostic_token_traces(steps, traces, "Test Traces", png3, csv3)

    assert png3.exists()
    assert csv3.exists()
