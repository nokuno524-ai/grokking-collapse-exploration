import numpy as np
import pytest
import sys
import os

# Add the project root to python path to import analysis
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from analysis.phase_transition import detect_phase_transitions, correlate_with_weight_norm

def test_detect_phase_transitions_crossover():
    """Test detection of grokking crossover point."""
    steps = np.arange(0, 1000, 10)

    # Train acc is high immediately
    train_acc = np.ones_like(steps) * 0.95

    # Test acc stays low then jumps
    test_acc = np.zeros_like(steps, dtype=float)
    test_acc[50:] = 0.95 # Jumps at index 50 (step 500)

    train_loss = np.zeros_like(steps, dtype=float)
    test_loss = np.ones_like(steps, dtype=float)

    results = detect_phase_transitions(steps, train_acc, test_acc, train_loss, test_loss)

    assert results['crossover_step'] == 500
    assert results['confidence_score'] > 0.0

def test_detect_phase_transitions_acc_jump():
    """Test detection of sudden accuracy jump."""
    steps = np.arange(0, 100, 1)

    train_acc = np.zeros_like(steps, dtype=float)
    test_acc = np.zeros_like(steps, dtype=float)

    # Gradual increase then sudden jump > 10%
    test_acc[0:20] = np.linspace(0, 0.1, 20)
    test_acc[20] = 0.3 # 20% jump at step 20
    test_acc[21:] = 0.3

    train_loss = np.zeros_like(steps, dtype=float)
    test_loss = np.zeros_like(steps, dtype=float)

    results = detect_phase_transitions(steps, train_acc, test_acc, train_loss, test_loss, acc_jump_threshold=0.10)

    assert results['acc_jump_step'] == 20

def test_detect_phase_transitions_short_array():
    """Test graceful handling of very short arrays."""
    steps = np.array([0, 1])
    train_acc = np.array([0.0, 0.1])
    test_acc = np.array([0.0, 0.1])
    train_loss = np.array([1.0, 0.9])
    test_loss = np.array([1.0, 0.9])

    results = detect_phase_transitions(steps, train_acc, test_acc, train_loss, test_loss)

    assert results['crossover_step'] is None
    assert results['acc_jump_step'] is None
    assert results['confidence_score'] == 0.0

def test_correlate_with_weight_norm():
    """Test correlation computation."""
    steps = np.arange(0, 100, 10)
    test_acc = np.linspace(0, 1, 10)
    # Perfectly correlated
    weight_norms = np.linspace(10, 20, 10)

    results = correlate_with_weight_norm(steps, test_acc, weight_norms, transition_step=50)

    assert 'pearson_correlation' in results
    assert np.isclose(results['pearson_correlation'], 1.0)
    assert np.isclose(results["wn_diff_to_transition"], 5.555555555555555)
    assert np.isclose(results["acc_diff_to_transition"], 0.5555555555555556)

def test_correlate_with_weight_norm_invalid_step():
    """Test correlation with invalid step."""
    steps = np.array([10, 20, 30])
    test_acc = np.array([0.1, 0.2, 0.3])
    weight_norms = np.array([1.0, 2.0, 3.0])

    results = correlate_with_weight_norm(steps, test_acc, weight_norms, transition_step=40)
    assert 'error' in results
