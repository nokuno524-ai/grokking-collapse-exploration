import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from scipy.signal import savgol_filter
import pandas as pd

def detect_phase_transitions(
    steps: np.ndarray,
    train_acc: np.ndarray,
    test_acc: np.ndarray,
    train_loss: np.ndarray,
    test_loss: np.ndarray,
    acc_jump_threshold: float = 0.10
) -> Dict[str, Any]:
    """
    Analyzes learning curves for grokking signatures and phase transitions.

    Args:
        steps: Array of training steps
        train_acc: Array of train accuracies
        test_acc: Array of test accuracies
        train_loss: Array of train losses
        test_loss: Array of test losses
        acc_jump_threshold: Threshold for sudden accuracy jump

    Returns:
        Dict with detected transitions, metrics, and confidence scores.
    """
    results = {
        'crossover_step': None,
        'acc_jump_step': None,
        'acceleration_step': None,
        'confidence_score': 0.0
    }

    if len(steps) < 3:
        return results

    # 1. Grokking signature: Crossover where test_acc catches up to train_acc
    # Find where train_acc is high (e.g. > 0.9) but test_acc is low (e.g. < 0.5)
    # Then find when test_acc finally crosses a threshold (e.g. 0.9)
    train_grokked = train_acc > 0.9
    test_grokked = test_acc > 0.9

    if np.any(test_grokked):
        first_test_grok = np.argmax(test_grokked)
        results['crossover_step'] = int(steps[first_test_grok])
        results['confidence_score'] += 0.4

    # 2. Sudden accuracy jump (>10% in a single step)
    # Calculate step-wise differences in test accuracy
    test_acc_diffs = np.diff(test_acc)
    jump_indices = np.where(test_acc_diffs > acc_jump_threshold)[0]

    if len(jump_indices) > 0:
        # Use the first major jump
        results['acc_jump_step'] = int(steps[jump_indices[0] + 1]) # +1 because diff array is 1 shorter
        results['confidence_score'] += 0.3

    # 3. Second derivative of test loss/accuracy to find acceleration
    # Use savgol_filter to smooth if array is long enough, otherwise just np.gradient
    if len(test_acc) >= 5:
        window_len = min(11, len(test_acc) if len(test_acc) % 2 != 0 else len(test_acc) - 1)
        if window_len >= 3:
            smoothed_acc = savgol_filter(test_acc, window_len, 2)
        else:
            smoothed_acc = test_acc
    else:
        smoothed_acc = test_acc

    first_deriv = np.gradient(smoothed_acc)
    second_deriv = np.gradient(first_deriv)

    # Find step with maximum acceleration (max second derivative)
    max_accel_idx = np.argmax(second_deriv)
    if second_deriv[max_accel_idx] > 0.01: # Small threshold to ensure it's a real acceleration
        results['acceleration_step'] = int(steps[max_accel_idx])
        results['confidence_score'] += 0.3

    return results

if __name__ == "__main__":
    pass

def correlate_with_weight_norm(
    steps: np.ndarray,
    test_acc: np.ndarray,
    weight_norms: np.ndarray,
    transition_step: int
) -> Dict[str, Any]:
    """
    Computes correlations between accuracy and weight norms around a phase transition.

    Args:
        steps: Array of training steps
        test_acc: Array of test accuracies
        weight_norms: Array of model weight norms
        transition_step: The step where phase transition occurred

    Returns:
        Dictionary of correlation metrics.
    """
    from scipy.stats import pearsonr, spearmanr

    if len(steps) < 3 or transition_step not in steps:
        return {'error': 'Not enough data or transition step not found'}

    # Get index of transition step
    trans_idx = np.where(steps == transition_step)[0][0]

    # Calculate difference in weight norm and accuracy from start to transition
    if trans_idx > 0:
        wn_diff_before = weight_norms[trans_idx] - weight_norms[0]
        acc_diff_before = test_acc[trans_idx] - test_acc[0]
    else:
        wn_diff_before = 0.0
        acc_diff_before = 0.0

    # Correlation for the whole trajectory
    try:
        pearson_corr, p_value_p = pearsonr(test_acc, weight_norms)
        spearman_corr, p_value_s = spearmanr(test_acc, weight_norms)
    except Exception:
        pearson_corr, p_value_p = 0.0, 1.0
        spearman_corr, p_value_s = 0.0, 1.0

    return {
        'wn_diff_to_transition': float(wn_diff_before),
        'acc_diff_to_transition': float(acc_diff_before),
        'pearson_correlation': float(pearson_corr),
        'spearman_correlation': float(spearman_corr),
        'p_value': float(p_value_p)
    }

def analyze_trajectory_file(trajectory_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Analyzes a single trajectory JSON file for phase transitions and correlates with weight norms.

    Args:
        trajectory_path: Path to trajectory JSON (e.g. results/pure/results.json)
        output_path: Optional path to save the phase transition results

    Returns:
        Dictionary of analysis results.
    """
    with open(trajectory_path, 'r') as f:
        data = json.load(f)

    history = data.get('history', [])
    if not history:
        return {'error': 'No history found in trajectory'}

    df = pd.DataFrame(history)

    # Extract arrays
    steps = df['step'].values
    train_acc = df.get('train_acc', np.zeros_like(steps)).values
    test_acc = df.get('test_acc', np.zeros_like(steps)).values
    train_loss = df.get('train_loss', np.zeros_like(steps)).values
    test_loss = df.get('test_loss', np.zeros_like(steps)).values
    weight_norms = df.get('weight_norm', np.zeros_like(steps)).values

    # Detect transitions
    transitions = detect_phase_transitions(steps, train_acc, test_acc, train_loss, test_loss)

    # Analyze correlations if we found a crossover step
    if transitions['crossover_step'] is not None:
        correlations = correlate_with_weight_norm(
            steps, test_acc, weight_norms, transitions['crossover_step']
        )
        transitions['correlations'] = correlations

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(transitions, f, indent=2)

    return transitions
