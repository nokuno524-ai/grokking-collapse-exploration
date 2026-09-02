import numpy as np
from src.analysis.early_warning.signals import compute_signals

def test_compute_signals_basic():
    """Test signal computation on a small synthetic history matching hand-derived values."""
    history = [
        {'step': 100, 'train_loss': 2.0, 'test_acc': 0.1, 'weight_norm': 10.0},
        {'step': 200, 'train_loss': 1.5, 'test_acc': 0.1, 'weight_norm': 15.0},
        {'step': 300, 'train_loss': 1.0, 'test_acc': 0.1, 'weight_norm': 20.0},
        {'step': 400, 'train_loss': 0.5, 'test_acc': 0.2, 'weight_norm': 25.0},
    ]

    # max_step=300 means only steps 100, 200, 300 are included.
    # window_steps=200 means we look at steps >= 100 (since max is 300).
    signals = compute_signals(history, max_step=300, window_steps=200)

    # Hand derived values:
    # steps = [100, 200, 300]
    # train_loss = [2.0, 1.5, 1.0] => slope = -0.005
    # weight_norm = [10, 15, 20] => slope = 0.05
    # test_acc = [0.1, 0.1, 0.1] => var = 0, autocorr = 0

    assert np.isclose(signals['train_loss_slope'], -0.005)
    assert np.isclose(signals['weight_norm_slope'], 0.05)
    assert np.isclose(signals['test_acc_var'], 0.0, atol=1e-10)
    assert np.isclose(signals['test_acc_autocorr'], 0.0)
    assert np.isclose(signals['delayed_gen_score'], 0.0)

def test_compute_signals_no_leakage():
    """Ensure signals are strictly computed from steps <= max_step."""
    history = [
        {'step': 100, 'train_loss': 2.0, 'test_acc': 0.1, 'weight_norm': 10.0},
        {'step': 200, 'train_loss': 1.5, 'test_acc': 0.1, 'weight_norm': 15.0},
        {'step': 300, 'train_loss': 1.0, 'test_acc': 0.1, 'weight_norm': 20.0},
        {'step': 400, 'train_loss': -10.0, 'test_acc': 1.0, 'weight_norm': 100.0}, # Huge jump at 400
    ]

    signals = compute_signals(history, max_step=300, window_steps=500)

    # Should not see the step 400 values at all
    assert np.isclose(signals['train_loss_slope'], -0.005)
    assert np.isclose(signals['weight_norm_slope'], 0.05)

def test_compute_signals_missing_keys():
    """Ensure it handles missing grad_norm gracefully."""
    history = [
        {'step': 100, 'train_loss': 2.0, 'test_acc': 0.1, 'weight_norm': 10.0},
        {'step': 200, 'train_loss': 1.5, 'test_acc': 0.1, 'weight_norm': 15.0},
    ]
    signals = compute_signals(history, max_step=200)
    assert np.isnan(signals['grad_norm_mean'])
    assert np.isnan(signals['grad_norm_var'])
    assert not np.isnan(signals['train_loss_slope'])

def test_delayed_gen_score_rising():
    """Test that rising variance and autocorrelation gives a positive delayed_gen_score."""
    history = [
        {'step': 100, 'train_loss': 2.0, 'test_acc': 0.1, 'weight_norm': 10.0},
        {'step': 200, 'train_loss': 1.5, 'test_acc': 0.2, 'weight_norm': 15.0},
        {'step': 300, 'train_loss': 1.0, 'test_acc': 0.4, 'weight_norm': 20.0},
        {'step': 400, 'train_loss': 0.5, 'test_acc': 0.5, 'weight_norm': 25.0},
    ]
    signals = compute_signals(history, max_step=400, window_steps=500)

    assert signals['test_acc_var'] > 0
    assert signals['test_acc_autocorr'] > 0
    assert signals['delayed_gen_score'] > 0
