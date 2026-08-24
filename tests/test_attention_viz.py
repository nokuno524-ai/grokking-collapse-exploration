import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from src.analysis.attention_viz import (
    compute_attention_entropy,
    compute_head_specialization,
    plot_attention_heatmap,
    plot_attention_entropy_trajectory,
    plot_head_specialization_trajectory
)
from src.model import ModularArithmeticTransformer
from scripts.extract_attention import get_all_layers_attention_weights

def test_extract_attention_weights():
    # Test with synthetic tiny transformer
    model = ModularArithmeticTransformer(d_model=32, n_heads=2, n_layers=2)
    model.eval()
    x = torch.randint(0, 59, (4, 2))

    with torch.no_grad():
        attn_weights = get_all_layers_attention_weights(model, x)

    assert len(attn_weights) == 2 # 2 layers
    for w in attn_weights:
        # B=4, n_heads=2, T=2, T=2
        assert w.shape == (4, 2, 2, 2)
        # Check sum to 1 over last dim
        torch.testing.assert_close(w.sum(dim=-1), torch.ones(4, 2, 2))

def test_compute_attention_entropy():
    # Shape: (B=1, n_heads=1, T=2, T=2)
    attn_uniform = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    ent = compute_attention_entropy(attn_uniform)
    expected_ent = torch.log(torch.tensor(2.0))
    torch.testing.assert_close(ent[0, 0, 0], expected_ent)

    attn_onehot = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    ent_onehot = compute_attention_entropy(attn_onehot)
    torch.testing.assert_close(ent_onehot[0, 0, 0], torch.tensor(0.0), atol=1e-5, rtol=1e-5)

def test_compute_head_specialization():
    attn_identical = torch.tensor([
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[1.0, 0.0], [0.5, 0.5]]
        ]
    ])
    spec = compute_head_specialization(attn_identical)
    assert torch.all(spec == 0)

    attn_diff = torch.tensor([
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.0, 1.0], [0.1, 0.9]]
        ]
    ])
    spec_diff = compute_head_specialization(attn_diff)
    assert torch.any(spec_diff > 0)

def test_plotting_functions_run_without_error():
    attn = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    attention_data = {
        "pure": {100: [attn], 200: [attn]},
        "collapse": {100: [attn], 200: [attn]}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        fig1 = plot_attention_heatmap([attn], 0, 0, 0)
        fig1.savefig(tmp_path / "heatmap.png")
        assert (tmp_path / "heatmap.png").exists()

        fig2 = plot_attention_entropy_trajectory(attention_data, layer_idx=0)
        fig2.savefig(tmp_path / "ent.png")
        assert (tmp_path / "ent.png").exists()

        fig3 = plot_head_specialization_trajectory(attention_data, layer_idx=0)
        fig3.savefig(tmp_path / "spec.png")
        assert (tmp_path / "spec.png").exists()
