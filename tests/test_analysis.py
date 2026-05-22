import pytest
import pandas as pd
from pathlib import Path
from analysis.analyze_results import build_dataframe, generate_summary_tables

def test_build_dataframe():
    mock_results = [
        {"condition": "pure", "seed": 1, "final_test_acc": 1.0, "history": [1, 2]},
        {"condition": "collapse", "seed": 2, "final_test_acc": 0.5, "history": [1, 2]},
    ]
    df = build_dataframe(mock_results)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "history" not in df.columns
    assert "final_test_acc" in df.columns

def test_generate_summary_tables(tmp_path):
    mock_results = [
        {"condition": "pure", "seed": 1, "final_test_acc": 1.0},
        {"condition": "pure", "seed": 2, "final_test_acc": 0.9},
        {"condition": "collapse", "seed": 1, "final_test_acc": 0.5},
    ]
    df = pd.DataFrame(mock_results)

    out_file = tmp_path / "summary.md"
    summary = generate_summary_tables(df, out_file)

    assert out_file.exists()

    # Check that pure condition mean is calculated correctly (0.95)
    assert "pure" in summary.index
    assert "collapse" in summary.index

    # For older pandas versions it might be multi-index, but we flattened it
    # We can just check the values in the DataFrame that is returned
    # The columns are flattened like 'final_test_acc_mean'
    if 'final_test_acc_mean' in summary.columns:
        assert summary.loc["pure", "final_test_acc_mean"] == 0.95
        assert summary.loc["collapse", "final_test_acc_mean"] == 0.5
    else:
        # Fallback for some pandas setups
        assert summary.loc["pure", ("final_test_acc", "mean")] == 0.95

from analysis.paper_figures import plot_main_results, plot_grokking_timing, plot_weight_fourier_dynamics

def test_plot_generation(tmp_path):
    mock_results = [
        {
            "condition": "pure",
            "seed": 1,
            "grokking_step": 100,
            "history": [
                {"step": 10, "test_acc": 0.5, "weight_norm": 1.0, "fourier_concentration": 0.1},
                {"step": 20, "test_acc": 0.9, "weight_norm": 2.0, "fourier_concentration": 0.2}
            ]
        },
        {
            "condition": "pure",
            "seed": 2,
            "grokking_step": 120,
            "history": [
                {"step": 10, "test_acc": 0.4, "weight_norm": 1.1, "fourier_concentration": 0.1},
                {"step": 20, "test_acc": 0.8, "weight_norm": 2.1, "fourier_concentration": 0.3}
            ]
        }
    ]
    df = pd.DataFrame(mock_results)

    # Test plot generation functions execute without error
    plot_main_results(df, tmp_path)
    assert (tmp_path / "fig1_main_results.png").exists()
    assert (tmp_path / "fig1_main_results.pdf").exists()

    plot_grokking_timing(df, tmp_path)
    assert (tmp_path / "fig2_grokking_timing.png").exists()
    assert (tmp_path / "fig2_grokking_timing.pdf").exists()

    plot_weight_fourier_dynamics(df, tmp_path)
    assert (tmp_path / "fig3_weight_fourier.png").exists()
    assert (tmp_path / "fig3_weight_fourier.pdf").exists()
