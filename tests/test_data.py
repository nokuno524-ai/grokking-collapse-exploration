import pytest
import torch
import numpy as np
from src.data import generate_modular_arithmetic, DatasetConfig, apply_collapse, apply_label_noise

def test_data_generation_shapes():
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_pairs = 59 * 59
    n_train = int(total_pairs * 0.3)
    n_test = total_pairs - n_train

    assert train_in.shape == (n_train, 2)
    assert train_tgt.shape == (n_train,)
    assert test_in.shape == (n_test, 2)
    assert test_tgt.shape == (n_test,)

def test_collapse_level():
    config = DatasetConfig(prime=11, train_fraction=1.0, collapse_level=0.5, collapse_severity=0.9, seed=42)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    # Calculate pure targets for comparison
    pure_targets = [(a + b) % 11 for a, b in train_in.tolist()]

    mismatches = sum(1 for t, pt in zip(train_tgt.tolist(), pure_targets) if t != pt)

    # We expect roughly 50% of targets to be replaced. Not all replacements will be different from the pure target,
    # but many will be.
    assert mismatches > 0

def test_label_noise():
    config = DatasetConfig(prime=11, train_fraction=1.0, collapse_level=0.0, noise_fraction=0.2, seed=42)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    pure_targets = [(a + b) % 11 for a, b in train_in.tolist()]
    mismatches = sum(1 for t, pt in zip(train_tgt.tolist(), pure_targets) if t != pt)

    # noise_fraction ensures corruption is observable
    assert mismatches == int(11*11*0.2)

def test_apply_collapse_missing_target_prob():
    # Test that the 1e-10 base probability works for missing targets
    rng = np.random.RandomState(42)
    # create a dataset where target '5' is missing completely
    targets = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    pairs = [(i, i) for i in range(10)]

    new_pairs, new_targets = apply_collapse(pairs, targets, prime=6, collapse_level=1.0, collapse_severity=0.5, rng=rng)

    # Target 5 should almost never appear
    assert 5 not in new_targets
