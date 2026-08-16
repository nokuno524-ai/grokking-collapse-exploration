import pytest
import numpy as np
import torch
import json
import os
from pathlib import Path

# Add project root to path for imports to work
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.weight_dynamics import compute_effective_rank
from analysis.stats import cohen_d, permutation_test
from analysis.attention_evolution import get_attention_entropies
from src.model import ModularArithmeticTransformer

def test_compute_effective_rank():
    # Rank of identity matrix should be 1.0 (entropy 0.0, e^0 = 1.0)
    # wait, identity matrix singular values are all 1s
    # if n=4, s = [1,1,1,1]
    # s_norm = [0.25, 0.25, 0.25, 0.25]
    # entropy = 4 * (-0.25 * log(0.25)) = log(4)
    # rank = e^(log(4)) = 4.0

    W_id = torch.eye(4)
    rank = compute_effective_rank(W_id)
    assert np.isclose(rank, 4.0, atol=1e-4)

    # Rank of zero matrix should be 0.0
    W_zero = torch.zeros((4, 4))
    rank_zero = compute_effective_rank(W_zero)
    assert np.isclose(rank_zero, 0.0, atol=1e-4)

def test_cohen_d():
    # Simple arrays where we know the difference
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 10])

    # mean_x = 3, var_x = 2.5
    # mean_y = 8, var_y = 2.5
    # pool_var = 2.5
    # diff = -5
    # d = -5 / sqrt(2.5) = -3.162

    d = cohen_d(x, y)
    assert np.isclose(d, -3.162, atol=1e-3)

def test_permutation_test():
    # Distinct distributions
    x = np.array([10, 11, 12, 10, 11])
    y = np.array([1, 2, 3, 1, 2])

    p = permutation_test(x, y, n_permutations=1000)
    # They should be very significantly different
    assert p < 0.05

    # Identical distributions
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])

    p = permutation_test(x, y, n_permutations=100)
    assert p > 0.05

def test_get_attention_entropies():
    # Test that we can get entropy and the dimension is correct
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=4, d_ff=64)
    probe_batch = torch.randint(0, 11, (16, 2))

    entropies = get_attention_entropies(model, probe_batch)

    assert entropies.shape == (4,)
    assert np.all(entropies >= 0)
    # Maximum entropy for sequence length 2 is log(2) ~ 0.693
    assert np.all(entropies <= np.log(2) + 1e-5)

def test_analysis_scripts_e2e(tmpdir):
    """Test that scripts run gracefully even with empty/minimal data."""
    # Create mock results directory structure
    base = Path(tmpdir) / "results"
    base.mkdir()

    cond_dir = base / "pure"
    cond_dir.mkdir()

    # Mock results.json
    mock_results = {
        "grokking_step": 1500,
        "grokked": True,
        "final_test_acc": 0.99,
        "history": [{"step": 100, "weight_norm": 10.0}, {"step": 200, "weight_norm": 6.0}],
        "config": {"collapse_severity": 0.0, "collapse_level": 0.0}
    }
    with open(cond_dir / "results.json", "w") as f:
        json.dump(mock_results, f)

    # Mock checkpoint
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    torch.save({
        "step": 200,
        "model_state": model.state_dict(),
        "config": {"prime": 59, "d_model": 32, "n_heads": 2, "d_ff": 64, "n_layers": 1}
    }, cond_dir / "checkpoint_200.pt")

    out_dir = Path(tmpdir) / "out"

    from analysis.attention_evolution import analyze_attention_evolution
    analyze_attention_evolution(base, out_dir / "attention", device="cpu")
    assert (out_dir / "attention" / "entropy_curves_pure.png").exists()

    from analysis.weight_dynamics import analyze_weight_dynamics
    analyze_weight_dynamics(base, out_dir / "weights")
    assert (out_dir / "weights" / "norm_traj_pure.png").exists()

    from analysis.phase_diagram import analyze_phase_diagram
    analyze_phase_diagram(base, out_dir / "phase")
    assert (out_dir / "phase" / "phase_diagram_accuracy.png").exists()

    from analysis.stats import compute_stats
    compute_stats(base, out_dir / "stats")
    assert (out_dir / "stats" / "summary_stats.csv").exists()
