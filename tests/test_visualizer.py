import os
import pathlib
import pytest
import pandas as pd
import numpy as np

from src.analysis.visualizer import (
    plot_loss_curves,
    plot_accuracy_curves,
    plot_collapse_comparison,
    plot_weight_norm_trajectory,
    plot_attention_heatmap,
    plot_attention_evolution,
    generate_experiment_report
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "step": [100, 200, 300, 400],
        "train_loss": [2.0, 1.5, 0.5, 0.1],
        "test_loss": [2.1, 1.6, 0.8, 0.2],
        "train_acc": [0.1, 0.4, 0.8, 0.99],
        "test_acc": [0.1, 0.3, 0.7, 0.96],
        "weight_norm": [10.0, 11.0, 8.0, 5.0]
    })

def test_plot_loss_curves(tmp_path, sample_df):
    out_path = tmp_path / "loss.png"
    plot_loss_curves(sample_df, str(out_path))
    assert out_path.exists()

def test_plot_accuracy_curves(tmp_path, sample_df):
    out_path = tmp_path / "acc.png"
    plot_accuracy_curves(sample_df, str(out_path))
    assert out_path.exists()

def test_plot_collapse_comparison(tmp_path, sample_df):
    runs = {
        "pure": sample_df,
        "collapse": sample_df.assign(test_acc=[0.1, 0.1, 0.2, 0.3])
    }
    plot_collapse_comparison(runs, str(tmp_path))
    assert (tmp_path / "collapse_comparison_acc.png").exists()
    assert (tmp_path / "collapse_comparison_norm.png").exists()

def test_plot_weight_norm_trajectory(tmp_path, sample_df):
    out_path = tmp_path / "norm.png"
    plot_weight_norm_trajectory(sample_df, str(out_path))
    assert out_path.exists()

def test_plot_attention_heatmap(tmp_path):
    attn = np.random.rand(10, 10)
    out_path = tmp_path / "attn.png"
    plot_attention_heatmap(attn, str(out_path))
    assert out_path.exists()

def test_plot_attention_evolution(tmp_path):
    snapshots = [np.random.rand(10, 10) for _ in range(3)]
    out_path = tmp_path / "attn_evol.png"
    plot_attention_evolution(snapshots, str(out_path))
    assert out_path.exists()

def test_generate_experiment_report(tmp_path):
    import json
    res_dir = tmp_path / "results"
    res_dir.mkdir()

    cond_dir = res_dir / "pure"
    cond_dir.mkdir()

    results_file = cond_dir / "results.json"
    results_file.write_text(json.dumps({
        "config": {"collapse_level": 0.0},
        "grokked": True,
        "final_test_acc": 0.96,
        "history": [
            {"step": 100, "train_loss": 2.0, "test_loss": 2.1, "train_acc": 0.1, "test_acc": 0.1, "weight_norm": 10.0},
            {"step": 400, "train_loss": 0.1, "test_loss": 0.2, "train_acc": 0.99, "test_acc": 0.96, "weight_norm": 5.0}
        ]
    }))

    out_dir = tmp_path / "report"
    generate_experiment_report(str(res_dir), str(out_dir))

    assert (out_dir / "index.html").exists()
    assert (out_dir / "collapse_comparison_acc.png").exists()
    assert (out_dir / "pure" / "loss.png").exists()
