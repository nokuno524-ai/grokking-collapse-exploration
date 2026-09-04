import pytest
import os
import torch
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.model import ModularArithmeticTransformer
from src.analysis.attention import extract_attention_weights
import scripts.visualize_attention as vz

def test_extract_attention_weights_no_mha():
    """Test error path when model has no MultiheadAttention."""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    inputs = torch.randn(1, 10)
    with pytest.raises(ValueError, match="Could not extract attention weights"):
        extract_attention_weights(model, inputs)

def test_get_dummy_batch():
    batch = vz.get_dummy_batch(prime=11, batch_size=5)
    assert batch.shape == (5, 2)
    assert batch.max() < 11
    assert batch.min() >= 0

def test_generate_markdown_gallery(tmp_path):
    # Dummy data
    csv1 = tmp_path / "test1.csv"
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(csv1, index=False)

    single = [{
        "name": "ckpt1",
        "heatmap_path": Path("heat1.png"),
        "csv_path": csv1
    }]

    diff = [{
        "name_a": "ckpt1",
        "name_b": "ckpt2",
        "diff_path": Path("diff.png")
    }]

    vz.generate_markdown_gallery(tmp_path, single, diff)

    md_file = tmp_path / "attention_gallery.md"
    assert md_file.exists()

    content = md_file.read_text()
    assert "ckpt1" in content
    assert "heat1.png" in content
    assert "ckpt1 vs ckpt2" in content
    assert "diff.png" in content
    # Checking table rendering
    assert "A" in content
    assert "B" in content

@patch('scripts.visualize_attention.load_checkpoint')
def test_main_script(mock_load, tmp_path):
    # Mock load checkpoint
    model = ModularArithmeticTransformer(prime=11, d_model=16, n_heads=2, d_ff=32, n_layers=1)
    config = {"prime": 11}
    mock_load.return_value = (model, config)

    # Create dummy checkpoint files
    ckpt1 = tmp_path / "ckpt1.pt"
    ckpt2 = tmp_path / "ckpt2.pt"
    ckpt1.touch()
    ckpt2.touch()

    test_args = [
        "visualize_attention.py",
        "--checkpoints", str(ckpt1), str(ckpt2),
        "--names", "A", "B",
        "--compare",
        "--output-dir", str(tmp_path),
        "--batch-size", "4"
    ]

    with patch('sys.argv', test_args):
        vz.main()

    # Check outputs
    assert (tmp_path / "A_attention.png").exists()
    assert (tmp_path / "B_attention.png").exists()
    assert (tmp_path / "diff_A_vs_B.png").exists()
    assert (tmp_path / "attention_gallery.md").exists()
    assert (tmp_path / "A_metrics.csv").exists()
    assert (tmp_path / "A_similarity.csv").exists()
