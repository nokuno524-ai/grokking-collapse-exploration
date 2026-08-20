import pytest
import torch
import numpy as np
from src.data import (
    DatasetConfig,
    generate_modular_arithmetic,
    apply_collapse,
    apply_label_noise,
    get_all_conditions
)

def test_generate_modular_arithmetic_pure():
    config = DatasetConfig(prime=7, train_fraction=0.5, collapse_level=0.0, noise_fraction=0.0, seed=42)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_pairs = 7 * 7
    n_train = int(round(total_pairs * 0.5))
    n_test = total_pairs - n_train

    assert train_in.shape == (n_train, 2)
    assert train_tgt.shape == (n_train,)
    assert test_in.shape == (n_test, 2)
    assert test_tgt.shape == (n_test,)

    # Check targets are correct (pure condition)
    for i in range(n_train):
        a, b = train_in[i].tolist()
        tgt = train_tgt[i].item()
        assert (a + b) % 7 == tgt

def test_apply_collapse():
    prime = 7
    pairs = [(a, b) for a in range(prime) for b in range(prime)]
    targets = [(a + b) % prime for a, b in pairs]
    rng = np.random.RandomState(42)

    collapse_level = 0.5
    collapse_severity = 0.9

    new_pairs, new_targets = apply_collapse(pairs, targets, prime, collapse_level, collapse_severity, rng)

    assert len(new_pairs) == len(pairs)
    assert len(new_targets) == len(targets)

    # At least some targets should have changed if severity is high
    changed = sum(1 for a, b in zip(targets, new_targets) if a != b)
    assert changed > 0

def test_apply_label_noise():
    prime = 7
    pairs = [(a, b) for a in range(prime) for b in range(prime)]
    targets = [(a + b) % prime for a, b in pairs]
    rng = np.random.RandomState(42)

    noise_fraction = 0.5

    new_pairs, new_targets = apply_label_noise(pairs, targets, prime, noise_fraction, rng)

    assert len(new_pairs) == len(pairs)
    assert len(new_targets) == len(targets)

    # Check exactly the correct number of targets changed
    expected_changed = int(round(len(targets) * noise_fraction))
    changed = sum(1 for a, b in zip(targets, new_targets) if a != b)
    assert changed == expected_changed

def test_get_all_conditions():
    conditions = get_all_conditions(prime=11, seed=123)
    assert isinstance(conditions, dict)
    assert "pure" in conditions
    assert "severe_collapse" in conditions

    pure_config = conditions["pure"]
    assert pure_config.prime == 11
    assert pure_config.seed == 123
    assert pure_config.collapse_level == 0.0
