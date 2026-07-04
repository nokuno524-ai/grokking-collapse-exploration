import torch
import pytest
import numpy as np
from src.data import DatasetConfig, generate_modular_arithmetic, apply_collapse, apply_label_noise

def test_generate_modular_arithmetic():
    config = DatasetConfig(prime=59, train_fraction=0.3)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    # 59 * 59 = 3481 total pairs
    # 3481 * 0.3 = 1044 train, 2437 test
    assert len(train_in) == 1044
    assert len(test_in) == 2437
    assert len(train_tgt) == 1044
    assert len(test_tgt) == 2437

    # Inputs should be (batch, 2)
    assert train_in.shape[1] == 2
    assert test_in.shape[1] == 2

    # Targets should be within [0, 59)
    assert train_tgt.max() < 59
    assert train_tgt.min() >= 0

    # Verify exact math for one point
    for i in range(10):
        a, b = train_in[i].tolist()
        expected = (a + b) % 59
        assert train_tgt[i].item() == expected

def test_apply_collapse():
    prime = 59
    # Create simple dataset of 100 identical targets (extreme case)
    targets = [5] * 50 + [10] * 50
    pairs = [(0, 5)] * 50 + [(0, 10)] * 50

    rng = np.random.RandomState(42)

    # Apply severe collapse to 100% of data
    new_pairs, new_targets = apply_collapse(
        pairs, targets, prime,
        collapse_level=1.0, collapse_severity=1.0, rng=rng
    )

    assert len(new_targets) == 100

    # With severity 1.0 (temp=0.1), it should aggressively favor 5 and 10
    # Missing targets should not error out (due to 1e-10 fix)
    unique_targets = set(new_targets)
    # Check that it didn't just randomly pick an unseen target
    # Given the high temperature, it should almost exclusively pick 5 or 10
    assert 5 in unique_targets
    assert 10 in unique_targets

    # Also test an empty replace list (0% level)
    new_pairs_0, new_targets_0 = apply_collapse(
        pairs, targets, prime,
        collapse_level=0.0, collapse_severity=0.5, rng=rng
    )
    assert new_targets_0 == targets

def test_apply_label_noise():
    prime = 11
    targets = [0, 1, 2, 3, 4]
    pairs = [(0,0)] * 5
    rng = np.random.RandomState(42)

    new_pairs, new_targets = apply_label_noise(
        pairs, targets, prime,
        noise_fraction=1.0, rng=rng
    )

    assert len(new_targets) == 5

    # Every target should be different from its original
    for orig, new in zip(targets, new_targets):
        assert orig != new
        assert 0 <= new < prime
