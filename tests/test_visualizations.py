import pytest
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to pythonpath (handled by pytest config usually, but to be safe)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualizations.plot_training_curves import plot_curves
from visualizations.plot_phase_diagram import plot_phase_diagram
from visualizations.plot_attention_heatmaps import plot_attention_heatmaps

@pytest.fixture
def empty_data():
    return {}

@pytest.fixture
def mock_results_data():
    return {
        "pure": {
            "config": {"collapse_severity": 0.0},
            "final_test_acc": 1.0,
            "history": [
                {"step": 100, "test_loss": 2.5, "test_acc": 0.1, "weight_norm": 10.0},
                {"step": 200, "test_loss": 0.5, "test_acc": 1.0, "weight_norm": 15.0}
            ]
        },
        "severe_collapse": {
            "config": {"collapse_severity": 1.0},
            "final_test_acc": 0.0,
            "history": [
                {"step": 100, "test_loss": 3.5, "test_acc": 0.0, "weight_norm": 5.0},
                {"step": 200, "test_loss": 3.5, "test_acc": 0.0, "weight_norm": 5.5}
            ]
        }
    }

def test_plot_curves_edge_cases(tmp_path, empty_data, mock_results_data):
    # Test empty data
    empty_out = tmp_path / "empty_curves.png"
    plot_curves(empty_data, output_path=str(empty_out))
    assert empty_out.exists()

    # Test valid data
    valid_out = tmp_path / "valid_curves.png"
    plot_curves(mock_results_data, output_path=str(valid_out))
    assert valid_out.exists()


def test_plot_phase_diagram_edge_cases(tmp_path, empty_data, mock_results_data):
    # Test empty data
    empty_out = tmp_path / "empty_phase.png"
    plot_phase_diagram(empty_data, output_path=str(empty_out))
    assert empty_out.exists()

    # Test single point
    single_point_data = {
        "pure": {"config": {"collapse_severity": 0.0}, "final_test_acc": 1.0}
    }
    single_out = tmp_path / "single_phase.png"
    plot_phase_diagram(single_point_data, output_path=str(single_out))
    assert single_out.exists()

    # Test valid data
    valid_out = tmp_path / "valid_phase.png"
    plot_phase_diagram(mock_results_data, output_path=str(valid_out))
    assert valid_out.exists()

def test_plot_attention_heatmaps_missing_checkpoint(tmp_path):
    # Test missing checkpoint
    out_path = tmp_path / "missing_checkpoint_heatmaps.png"
    plot_attention_heatmaps("non_existent_checkpoint.pt", output_path=str(out_path))
    assert out_path.exists()
