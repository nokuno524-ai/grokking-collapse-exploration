import torch
import pytest
import numpy as np
from src.data import generate_modular_arithmetic, DatasetConfig, apply_collapse

def test_generate_modular_arithmetic():
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_pairs = 59 * 59
    n_train = int(total_pairs * 0.3)
    n_test = total_pairs - n_train

    assert train_in.shape == (n_train, 2)
    assert train_tgt.shape == (n_train,)
    assert test_in.shape == (n_test, 2)
    assert test_tgt.shape == (n_test,)

    # Check that without collapse, targets are exactly (a+b)%59
    expected_train_tgt = (train_in[:, 0] + train_in[:, 1]) % 59
    assert torch.all(train_tgt == expected_train_tgt)

    expected_test_tgt = (test_in[:, 0] + test_in[:, 1]) % 59
    assert torch.all(test_tgt == expected_test_tgt)

def test_apply_collapse():
    prime = 5
    # Create simple sequence: pairs (0,0)... and targets 0, 1, 2, 3, 4
    pairs = [(0, 0)] * 100
    # Targets mostly 0 to simulate peaked distribution
    targets = [0] * 60 + [1] * 20 + [2] * 10 + [3] * 10
    # 4 is entirely missing

    rng = np.random.RandomState(42)
    new_pairs, new_targets = apply_collapse(pairs, targets, prime, collapse_level=0.5, collapse_severity=0.5, rng=rng)

    assert len(new_pairs) == 100
    assert len(new_targets) == 100

    # Target 4 should not appear in collapsed targets (because base_prob is 0.0)
    assert 4 not in new_targets

    # Because collapse_level is 0.5, ~50 items were replaced. The remaining 50 should be unchanged.
    # The original targets had no 4, and the replaced ones also have no 4.
