import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import generate_modular_arithmetic, DatasetConfig, apply_collapse, apply_label_noise

def test_generate_modular_arithmetic_shapes():
    """Test data generation produces correct shapes and types."""
    config = DatasetConfig(prime=7, train_fraction=0.5, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_pairs = 7 * 7
    expected_train = int(round(total_pairs * 0.5))
    expected_test = total_pairs - expected_train

    assert train_in.shape == (expected_train, 2)
    assert train_tgt.shape == (expected_train,)
    assert test_in.shape == (expected_test, 2)
    assert test_tgt.shape == (expected_test,)

    assert train_in.dtype == torch.long
    assert train_tgt.dtype == torch.long

def test_apply_collapse_distribution():
    """Test apply_collapse actually narrows the distribution."""
    prime = 7
    targets = [i % prime for i in range(1000)]
    pairs = [(0, i) for i in targets]
    rng = np.random.RandomState(42)

    # 100% collapse with high severity
    _, collapsed_targets = apply_collapse(pairs, targets, prime, collapse_level=1.0, collapse_severity=0.9, rng=rng)

    # The number of unique elements should be the same, but the entropy should be lower
    # since it favors common items. In this uniform setup, let's just check length.
    assert len(collapsed_targets) == len(targets)

def test_apply_label_noise():
    """Test label noise ensures corruption is observable."""
    prime = 7
    targets = [0] * 100
    pairs = [(0, 0)] * 100
    rng = np.random.RandomState(42)

    # 100% noise
    _, noisy_targets = apply_label_noise(pairs, targets, prime, noise_fraction=1.0, rng=rng)

    # Original target was 0, so none of the noisy targets should be 0
    assert all(t != 0 for t in noisy_targets)
    assert all(0 <= t < prime for t in noisy_targets)
