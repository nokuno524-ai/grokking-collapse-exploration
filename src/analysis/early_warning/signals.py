"""
Signal extraction module for computing early warning indicators
from training logs prior to a given step.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple

def compute_signals(history: List[Dict], max_step: int, window_steps: int = 500) -> Dict[str, float]:
    """
    Computes early warning signals from a training log history up to max_step.

    Args:
        history: List of dictionaries, each containing metrics at a given step.
                 Expected keys: 'step', 'train_loss', 'test_acc', 'weight_norm'.
                 Optional keys: 'grad_norm'.
        max_step: Only use data where step <= max_step (strictly no leakage).
        window_steps: The size of the rolling window (in steps, not number of data points)
                      to compute slopes, variances, etc.

    Returns:
        Dictionary of computed signals.
    """
    # Filter history up to max_step
    filtered_history = [h for h in history if h['step'] <= max_step]
    if len(filtered_history) == 0:
        return {}

    # Extract arrays
    steps = np.array([h['step'] for h in filtered_history])
    train_loss = np.array([h.get('train_loss', np.nan) for h in filtered_history])
    test_acc = np.array([h.get('test_acc', np.nan) for h in filtered_history])
    weight_norm = np.array([h.get('weight_norm', np.nan) for h in filtered_history])
    grad_norm = np.array([h.get('grad_norm', np.nan) for h in filtered_history])

    # Filter to recent window
    recent_mask = steps >= max_step - window_steps
    recent_steps = steps[recent_mask]
    recent_train_loss = train_loss[recent_mask]
    recent_test_acc = test_acc[recent_mask]
    recent_weight_norm = weight_norm[recent_mask]

    signals = {}

    # 1. Train loss plateau slope
    valid = ~np.isnan(recent_train_loss)
    if len(recent_steps[valid]) > 1:
        # Fit y = mx + c
        m, _ = np.polyfit(recent_steps[valid], recent_train_loss[valid], 1)
        signals['train_loss_slope'] = float(m)
    else:
        signals['train_loss_slope'] = np.nan

    # 2. Weight norm derivative
    valid = ~np.isnan(recent_weight_norm)
    if len(recent_steps[valid]) > 1:
        m, _ = np.polyfit(recent_steps[valid], recent_weight_norm[valid], 1)
        signals['weight_norm_slope'] = float(m)
    else:
        signals['weight_norm_slope'] = np.nan

    # 3. Gradient norm statistics
    recent_grad_norm = grad_norm[recent_mask]
    if len(recent_grad_norm) > 0 and not np.all(np.isnan(recent_grad_norm)):
        valid_grads = recent_grad_norm[~np.isnan(recent_grad_norm)]
        if len(valid_grads) > 0:
            signals['grad_norm_mean'] = float(np.mean(valid_grads))
            signals['grad_norm_var'] = float(np.var(valid_grads))
        else:
            signals['grad_norm_mean'] = np.nan
            signals['grad_norm_var'] = np.nan
    else:
        signals['grad_norm_mean'] = np.nan
        signals['grad_norm_var'] = np.nan

    # 4. Test accuracy variance and autocorrelation
    if len(recent_test_acc) > 1 and not np.all(np.isnan(recent_test_acc)):
        valid_acc = recent_test_acc[~np.isnan(recent_test_acc)]
        if len(valid_acc) > 1:
            acc_var = float(np.var(valid_acc))
            signals['test_acc_var'] = acc_var

            # Lag-1 autocorrelation
            if len(valid_acc) > 2 and acc_var > 1e-12:
                mu = float(np.mean(valid_acc))
                x = valid_acc - mu
                autocorr = float(np.sum(x[:-1] * x[1:]) / (np.sum(x**2) + 1e-12))
                signals['test_acc_autocorr'] = autocorr
            else:
                signals['test_acc_autocorr'] = 0.0
        else:
            signals['test_acc_var'] = np.nan
            signals['test_acc_autocorr'] = np.nan
    else:
        signals['test_acc_var'] = np.nan
        signals['test_acc_autocorr'] = np.nan

    # 5. Delayed generalization score (variance * autocorrelation)
    # Inspired by critical transitions literature: rising var + rising autocorr
    var = signals.get('test_acc_var', np.nan)
    autocorr = signals.get('test_acc_autocorr', np.nan)
    if not np.isnan(var) and not np.isnan(autocorr):
        signals['delayed_gen_score'] = float(var * max(0, autocorr))
    else:
        signals['delayed_gen_score'] = np.nan

    return signals
