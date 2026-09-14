import pytest
import torch
import torch.nn as nn
from src.attention_extraction import extract_attention
from scripts.compare_attention import get_attention_stats

def test_extract_attention_shapes():
    layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
    model = nn.TransformerEncoder(layer, num_layers=2)
    x = torch.randn(4, 3, 16)
    with extract_attention(model) as attns:
        _ = model(x)
    assert len(attns) == 2
    assert attns[0].shape == (4, 2, 3, 3)

def test_attention_stats():
    # uniform
    uniform = torch.ones(3, 3) / 3
    stats_uni = get_attention_stats(uniform)
    assert pytest.approx(stats_uni["kl_uniform"], abs=1e-5) == 0

    # one-hot
    one_hot = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
    stats_oh = get_attention_stats(one_hot)
    assert pytest.approx(stats_oh["entropy"], abs=1e-5) == 0
    assert pytest.approx(stats_oh["diag_dominance"]) == 1.0

import tempfile
from pathlib import Path
from scripts.aggregate_report import generate_html_report

def test_report_generation():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        # Create some dummy data
        res_dir = temp_dir / "results"
        cond_dir = res_dir / "pure"
        run_dir = cond_dir / "run1"
        run_dir.mkdir(parents=True)

        # Dummy stats
        import pandas as pd
        df = pd.DataFrame({
            "step": [0, 100],
            "layer": [0, 0],
            "head": [0, 0],
            "entropy": [1.0, 0.5],
            "diag_dominance": [0.1, 0.9],
            "kl_uniform": [0.1, 1.0]
        })
        df.to_csv(run_dir / "attention_stats.csv", index=False)

        # Dummy image
        import numpy as np
        import matplotlib.pyplot as plt
        plt.plot(np.random.randn(10))
        plt.savefig(run_dir / "attention_trajectories.png")
        plt.close()

        out_file = temp_dir / "report.html"
        generate_html_report(res_dir, out_file)

        assert out_file.exists()
        content = out_file.read_text()
        assert "Attention Analysis Report" in content
        assert "pure" in content
        assert "run1" in content
        assert "aggregated_attention_stats.csv" in [p.name for p in temp_dir.iterdir()]
