import pytest
import torch
import numpy as np
from src.data import generate_modular_arithmetic, DatasetConfig, apply_collapse

def test_data_generation_shapes():
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_samples = 59 * 59
    n_train = int(total_samples * 0.3)
    n_test = total_samples - n_train

    assert train_in.shape == (n_train, 2)
    assert train_tgt.shape == (n_train,)
    assert test_in.shape == (n_test, 2)
    assert test_tgt.shape == (n_test,)

def test_data_generation_deterministic():
    config1 = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0, seed=42)
    config2 = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0, seed=42)

    train_in1, train_tgt1, test_in1, test_tgt1 = generate_modular_arithmetic(config1)
    train_in2, train_tgt2, test_in2, test_tgt2 = generate_modular_arithmetic(config2)

    assert torch.equal(train_in1, train_in2)
    assert torch.equal(train_tgt1, train_tgt2)
    assert torch.equal(test_in1, test_in2)
    assert torch.equal(test_tgt1, test_tgt2)

def test_collapse_distribution():
    prime = 5
    rng = np.random.RandomState(42)
    # Creating a dummy target list missing some values (e.g. missing 4)
    pairs = [(0, 0)] * 100
    targets = [0] * 50 + [1] * 25 + [2] * 15 + [3] * 10

    # High severity should amplify common targets and missing targets (like 4)
    # will use the 1e-10 base probability instead of 1.0/prime
    new_pairs, new_targets = apply_collapse(
        pairs, targets, prime, collapse_level=1.0, collapse_severity=0.9, rng=rng
    )

    # 4 should practically never be sampled because its base probability was 1e-10
    assert 4 not in new_targets

    # It should have replaced all targets
    assert len(new_targets) == len(targets)
