import os
import json
import torch
import pytest
import shutil
import pandas as pd
from pathlib import Path

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModularArithmeticTransformer
from analysis.attention_evolution import identify_phase_transition, extract_attention_patterns
from analysis.circuit_formation import load_model
from analysis.aggregate_results import aggregate_results


@pytest.fixture
def mock_results_dir(tmp_path):
    """Create a mock results directory with json and checkpoint files."""
    base_dir = tmp_path / "results"
    base_dir.mkdir()

    # Create Pure condition
    pure_dir = base_dir / "pure"
    pure_dir.mkdir()

    res_data = {
        "grokked": True,
        "grokking_step": 1500,
        "final_test_acc": 1.0,
        "final_train_acc": 1.0,
        "final_weight_norm": 30.5,
        "history": [
            {"step": 500, "train_acc": 0.5, "test_acc": 0.1, "weight_norm": 20.0, "embedding_rank": 15.0, "fourier_concentration": 0.1},
            {"step": 1000, "train_acc": 1.0, "test_acc": 0.2, "weight_norm": 25.0, "embedding_rank": 20.0, "fourier_concentration": 0.2},
            {"step": 1500, "train_acc": 1.0, "test_acc": 0.95, "weight_norm": 30.0, "embedding_rank": 25.0, "fourier_concentration": 0.4},
            {"step": 2000, "train_acc": 1.0, "test_acc": 1.0, "weight_norm": 30.5, "embedding_rank": 25.5, "fourier_concentration": 0.5},
        ]
    }

    with open(pure_dir / "results.json", "w") as f:
        json.dump(res_data, f)

    # Create mock model checkpoint
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    ckpt = {
        "step": 2000,
        "model_state": model.state_dict(),
        "config": {"prime": 59, "d_model": 32, "n_heads": 2, "d_ff": 64}
    }
    torch.save(ckpt, pure_dir / "checkpoint_2000.pt")

    return base_dir

def test_identify_phase_transition(mock_results_dir):
    json_path = mock_results_dir / "pure" / "results.json"
    grok_step = identify_phase_transition(str(json_path))
    # the max diff is from step 1000 to 1500 (0.2 -> 0.95, diff = +0.75)
    # transition step should be 1500
    assert grok_step == 1500

def test_extract_attention_patterns(mock_results_dir):
    ckpt_path = mock_results_dir / "pure" / "checkpoint_2000.pt"
    model = load_model(str(ckpt_path))

    # dummy input
    x = torch.tensor([[10, 20], [5, 15]])
    attn = extract_attention_patterns(model, x)

    # output should be (batch_size, n_heads, seq_len, seq_len)
    assert attn.shape == (2, 2, 2, 2)

    # Check that attention distributions sum to 1
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 2, 2))

def test_aggregate_results(mock_results_dir):
    latex = aggregate_results(str(mock_results_dir))
    assert "Pure Data" in latex
    assert "100%" in latex
    assert "100.0%" in latex
