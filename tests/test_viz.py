import pytest
import torch
from pathlib import Path
from src.viz.loss_landscape import plot_loss_landscape
from src.viz.feature_activation import plot_feature_activations
from src.viz.accuracy_curves import plot_accuracy_curves

def test_loss_landscape(tmp_path):
    h_p = [{"step": 1, "train_loss": 2.0}, {"step": 2, "train_loss": 0.5}]
    h_c = [{"step": 1, "train_loss": 2.0}, {"step": 2, "train_loss": 1.5}]
    plot_loss_landscape(h_p, h_c, tmp_path)
    assert (tmp_path / "loss_landscape_geometry.png").exists()

def test_feature_activations(tmp_path):
    acts = {"pure": torch.randn(100), "collapse": torch.randn(100) + 2}
    plot_feature_activations(acts, tmp_path)
    assert (tmp_path / "feature_activations.png").exists()

def test_accuracy_curves(tmp_path):
    hists = {
        "pure": [{"step": 1, "test_acc": 0.5}, {"step": 2, "test_acc": 0.98}],
        "collapse": [{"step": 1, "test_acc": 0.5}, {"step": 2, "test_acc": 0.6}]
    }
    plot_accuracy_curves(hists, tmp_path)
    assert (tmp_path / "accuracy_curves.png").exists()
