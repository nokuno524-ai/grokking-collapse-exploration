import torch
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ModularArithmeticTransformer
from src.attention_extraction import extract_attention

def test_extract_attention_shapes():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64, n_layers=2)
    model.eval()
    x = torch.randint(0, 11, (4, 2))

    with extract_attention() as attentions:
        out = model(x)

    assert len(attentions) == 2, "Expected 2 layers of attention"
    for attn in attentions:
        assert attn.shape == (4, 2, 2, 2), "Expected shape (batch, n_heads, seq_len, seq_len)"

def test_attention_entropy():
    # uniform distribution
    attn_weights = np.ones((1, 1, 2, 2)) * 0.5
    from scripts.vis_attention import compute_entropy
    entropy = compute_entropy(attn_weights)
    assert np.allclose(entropy, np.log(2)), f"Expected {np.log(2)}, got {entropy}"

def test_fourier_probe():
    from scripts.vis_attention import fourier_probe
    # test periodic
    attn_weights = np.zeros((4, 4))
    attn_weights[0, 0] = 1
    attn_weights[2, 2] = 1
    score = fourier_probe(attn_weights)
    assert score > 0.0

def test_head_ablation_ranking(tmp_path):
    from scripts.vis_attention import rank_heads_by_ablation

    model = ModularArithmeticTransformer(prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=1)
    # create dummy dataloaders
    x = torch.randint(0, 5, (10, 2))
    y = (x[:, 0] + x[:, 1]) % 5
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    out_file = tmp_path / "ablation_rank.md"
    rank_heads_by_ablation(model, loader, loader, str(out_file))

    assert out_file.exists()
    content = out_file.read_text()
    assert "Head Ablation Ranking" in content
    assert "| 1 |" in content  # at least one rank exists

def test_figures_smoke(tmp_path):
    from scripts.vis_attention import plot_attention_heatmaps, plot_fourier_scores

    # Fake attentions: (n_layers=2, batch=1, n_heads=2, seq_len=2, seq_len=2)
    attns = np.random.rand(2, 1, 2, 2, 2)

    hm_path = tmp_path / "heatmap.png"
    plot_attention_heatmaps(attns, str(hm_path))
    assert hm_path.exists()

    fourier_path = tmp_path / "fourier.png"
    plot_fourier_scores(attns, str(fourier_path))
    assert fourier_path.exists()
