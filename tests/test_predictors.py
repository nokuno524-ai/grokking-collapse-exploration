import pytest
import numpy as np
from analysis import predictors

@pytest.fixture
def dummy_run_data_early():
    # Simulate a run with 4 steps in the pre-grok window
    return {
        "history": [
            {"step": 100, "train_loss": 3.5, "test_acc": 0.05, "weight_norm": 10.0},
            {"step": 200, "train_loss": 3.4, "test_acc": 0.06, "weight_norm": 12.0},
            {"step": 300, "train_loss": 3.3, "test_acc": 0.05, "weight_norm": 14.0},
            {"step": 400, "train_loss": 3.1, "test_acc": 0.07, "weight_norm": 16.0},
            {"step": 2000, "train_loss": 0.1, "test_acc": 1.0, "weight_norm": 20.0} # beyond pre_grok window
        ]
    }

def test_compute_early_warning_metrics(dummy_run_data_early):
    early = predictors.compute_early_warning_metrics(dummy_run_data_early, pre_grok_steps=1000)
    assert early is not None
    assert 'wn_slope' in early
    assert early['wn_slope'] > 0 # Weight norm goes 10->12->14->16
    assert 'loss_curvature' in early
    assert 'acc_variance' in early
    assert early['acc_variance'] > 0

def test_train_predictor_and_evaluate():
    # Synthetic metrics
    runs_metrics = [
        {'grok_success': True},
        {'grok_success': True},
        {'grok_success': False},
        {'grok_success': False}
    ]
    early_metrics_list = [
        {'wn_slope': 1.0, 'loss_curvature': -0.1, 'acc_variance': 0.02},
        {'wn_slope': 1.2, 'loss_curvature': -0.2, 'acc_variance': 0.01},
        {'wn_slope': -0.5, 'loss_curvature': 0.1, 'acc_variance': 0.00},
        {'wn_slope': -1.0, 'loss_curvature': 0.2, 'acc_variance': 0.00}
    ]

    clf, auroc, y_true, y_scores = predictors.train_predictor_and_evaluate(runs_metrics, early_metrics_list)
    assert clf is not None
    assert 0.0 <= auroc <= 1.0
    assert len(y_true) == 4
    assert len(y_scores) == 4
