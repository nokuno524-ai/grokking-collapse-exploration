import pytest
import numpy as np
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

# Adding the root dir to sys.path so we can import `analysis`
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.stats import mean_ci, bootstrap_ci, permutation_test
from analysis.registry import build_registry
from src.paper_figures.fig1_grok_curves import generate_grok_curves
from src.paper_figures.fig2_weight_norm import generate_weight_norm_curves
from src.paper_figures.fig3_cliff import generate_cliff_figure
from src.paper_figures.fig4_gap import generate_gap_figure
from src.paper_figures.fig5_combined import generate_combined_figure

# --- Test Stats ---

def test_mean_ci():
    data = [1, 2, 3, 4, 5]
    m, lower, upper = mean_ci(data)
    assert np.isclose(m, 3.0)
    assert lower < m < upper

def test_mean_ci_empty():
    m, lower, upper = mean_ci([])
    assert np.isnan(m)

def test_bootstrap_ci():
    np.random.seed(42)
    data = [1.0] * 50 + [2.0] * 50
    m, lower, upper = bootstrap_ci(data, num_bootstraps=100)
    assert np.isclose(m, 1.5)
    assert lower <= m <= upper

def test_permutation_test():
    np.random.seed(42)
    # Distinct groups should yield low p-value
    group_a = np.random.normal(0, 1, 100)
    group_b = np.random.normal(5, 1, 100)
    p_diff = permutation_test(group_a, group_b, num_permutations=100)
    assert p_diff < 0.05

    # Same groups should yield high p-value
    group_c = np.random.normal(0, 1, 100)
    p_same = permutation_test(group_a, group_c, num_permutations=100)
    assert p_same > 0.05

# --- Test Registry and Figures (Smoke tests) ---

@pytest.fixture
def mock_results_dir():
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create a mock run
        run_dir = tmp_path / "mock_run"
        run_dir.mkdir()

        mock_data = {
            "config": {
                "seed": 42,
                "weight_decay": 1.0,
                "condition_name": "pure",
                "noise_fraction": 0.0,
                "train_fraction": 0.3
            },
            "grokked": True,
            "grokking_step": 1000,
            "final_test_acc": 1.0,
            "history": [
                {"step": 100, "train_acc": 0.5, "test_acc": 0.1, "weight_norm": 10.0, "train_loss": 3.0, "test_loss": 4.0},
                {"step": 1000, "train_acc": 1.0, "test_acc": 0.99, "weight_norm": 20.0, "train_loss": 0.1, "test_loss": 0.2}
            ]
        }

        with open(run_dir / "results.json", "w") as f:
            json.dump(mock_data, f)

        yield tmp_path

def test_build_registry(mock_results_dir):
    out_file = mock_results_dir / "registry.json"
    registry = build_registry(mock_results_dir, out_file)

    assert len(registry) == 1
    assert registry[0]["seed"] == 42
    assert registry[0]["grokked"] is True
    assert out_file.exists()

def test_figures_smoke(mock_results_dir):
    out_file = mock_results_dir / "registry.json"
    # To mock the exp_c_grid check in fig3/fig5
    mock_data = build_registry(mock_results_dir, out_file)
    mock_data[0]["run_path"] = "exp_c_grid" # Mock path so it passes condition filtering

    with open(out_file, "w") as f:
        json.dump(mock_data, f)

    fig_dir = mock_results_dir / "figures"
    fig_dir.mkdir()

    # Should run without raising exceptions
    generate_grok_curves(out_file, fig_dir)
    generate_weight_norm_curves(out_file, fig_dir)
    generate_cliff_figure(out_file, fig_dir)
    generate_gap_figure(out_file, fig_dir)
    generate_combined_figure(out_file, fig_dir)

    assert (fig_dir / "fig1_grok_curves.pdf").exists()
    assert (fig_dir / "fig2_weight_norm.pdf").exists()
    assert (fig_dir / "fig3_cliff.pdf").exists()
    assert (fig_dir / "fig4_gap.pdf").exists()
    assert (fig_dir / "fig5_combined.pdf").exists()
