import pytest
import pandas as pd
from src.grokking.run_experiments import ExperimentRunConfig

def test_config_validation_success():
    config_data = {
        "model_size": {"d_model": 128, "n_heads": 4, "n_layers": 1},
        "dataset": {"prime": 59, "train_fraction": 0.3},
        "composition_ratios": [0.0, 0.5],
        "collapse_severities": [0.1, 0.9],
        "training_steps": 1000,
        "seeds": [1, 2],
        "output_dir": "test"
    }
    config = ExperimentRunConfig(**config_data)
    assert config.training_steps == 1000
    assert len(config.composition_ratios) == 2

def test_config_validation_failure():
    config_data = {
        "model_size": {"d_model": 128, "n_heads": 4, "n_layers": 1},
        "dataset": {"prime": 59, "train_fraction": 0.3},
        "composition_ratios": [1.5], # Invalid
        "collapse_severities": [0.1],
        "training_steps": 1000,
        "seeds": [1],
        "output_dir": "test"
    }
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        ExperimentRunConfig(**config_data)

def test_config_negative_steps():
    config_data = {
        "model_size": {"d_model": 128, "n_heads": 4, "n_layers": 1},
        "dataset": {"prime": 59, "train_fraction": 0.3},
        "composition_ratios": [0.5],
        "collapse_severities": [0.1],
        "training_steps": -10, # Invalid
        "seeds": [1],
        "output_dir": "test"
    }
    with pytest.raises(ValueError):
        ExperimentRunConfig(**config_data)

from src.grokking.aggregate_results import generate_summary_table

def test_aggregate_summary_table(tmp_path):
    data = [
        {"collapse_level": 0.0, "collapse_severity": 0.3, "final_train_acc": 0.9, "final_test_acc": 0.8, "final_weight_norm": 10.0, "final_embedding_rank": 5.0, "final_fourier_concentration": 0.2},
        {"collapse_level": 0.0, "collapse_severity": 0.3, "final_train_acc": 0.95, "final_test_acc": 0.85, "final_weight_norm": 12.0, "final_embedding_rank": 6.0, "final_fourier_concentration": 0.3},
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / "summary.csv"
    summary_df = generate_summary_table(df, str(csv_path))

    assert csv_path.exists()
    assert len(summary_df) == 1
    assert summary_df["collapse_level"].iloc[0] == 0.0
    assert summary_df["collapse_severity"].iloc[0] == 0.3
    assert summary_df["final_train_acc_mean"].iloc[0] == 0.925
