import os
import json
import torch
import pytest
from pathlib import Path
from viz.plot_training_curves import plot_training_curves
from viz.plot_weight_analysis import plot_weight_analysis

@pytest.fixture
def synthetic_data_dir(tmp_path):
    results_dir = tmp_path / "results"

    # Create pure and low_collapse data
    for condition in ["pure", "low_collapse"]:
        cond_dir = results_dir / condition
        cond_dir.mkdir(parents=True)

        # Mock results.json
        mock_data = {
            "config": {"condition_name": condition},
            "history": [
                {"step": 100, "test_acc": 0.5, "test_loss": 2.0},
                {"step": 200, "test_acc": 0.9, "test_loss": 0.5}
            ]
        }
        with open(cond_dir / "results.json", 'w') as f:
            json.dump(mock_data, f)

        # Mock checkpoints
        for step in [100, 200]:
            mock_ckpt = {
                "model_state": {
                    "layer1.weight": torch.ones((5, 5)) * step
                }
            }
            torch.save(mock_ckpt, cond_dir / f"checkpoint_{step}.pt")

    return results_dir

def test_plot_training_curves(synthetic_data_dir, tmp_path):
    output_dir = tmp_path / "output"

    # Run the function
    plot_training_curves(results_dir=str(synthetic_data_dir), output_dir=str(output_dir))

    # Check if the plot was generated
    assert (output_dir / "training_curves.png").exists()

def test_plot_weight_analysis(synthetic_data_dir, tmp_path):
    output_dir = tmp_path / "output"

    # Run the function
    plot_weight_analysis(results_dir=str(synthetic_data_dir), output_dir=str(output_dir))

    # Check if the plot was generated
    assert (output_dir / "weight_analysis.png").exists()
