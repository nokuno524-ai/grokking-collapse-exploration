import pytest
import torch
import numpy as np
from src.management.config import DatasetConfig
from src.data import generate_modular_arithmetic, apply_collapse, apply_label_noise, get_all_conditions

def test_generate_modular_arithmetic_shapes():
    config = DatasetConfig(prime=7, train_fraction=0.5)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_pairs = 7 * 7
    n_train = int(total_pairs * 0.5)
    n_test = total_pairs - n_train

    assert train_in.shape == (n_train, 2)
    assert train_tgt.shape == (n_train,)
    assert test_in.shape == (n_test, 2)
    assert test_tgt.shape == (n_test,)

def test_generate_modular_arithmetic_values():
    config = DatasetConfig(prime=7, train_fraction=1.0)
    train_in, train_tgt, _, _ = generate_modular_arithmetic(config)

    # Check if (a + b) mod p is correct
    a = train_in[:, 0]
    b = train_in[:, 1]
    expected_tgt = (a + b) % 7
    assert torch.equal(train_tgt, expected_tgt)

def test_apply_collapse():
    config = DatasetConfig(prime=5, train_fraction=1.0, collapse_level=0.5, collapse_severity=1.0, seed=42)
    train_in, train_tgt, _, _ = generate_modular_arithmetic(config)

    a = train_in[:, 0]
    b = train_in[:, 1]
    expected_clean_tgt = (a + b) % 5

    # Not all targets should match the clean targets due to collapse
    assert not torch.equal(train_tgt, expected_clean_tgt)

def test_apply_label_noise():
    config = DatasetConfig(prime=5, train_fraction=1.0, noise_fraction=1.0, seed=42)
    train_in, train_tgt, _, _ = generate_modular_arithmetic(config)

    a = train_in[:, 0]
    b = train_in[:, 1]
    expected_clean_tgt = (a + b) % 5

    # Every target should be different from clean due to 1.0 noise fraction
    assert torch.all(train_tgt != expected_clean_tgt)

def test_get_all_conditions():
    conditions = get_all_conditions(prime=11)
    assert "pure" in conditions
    assert "severe_collapse" in conditions
    assert conditions["pure"].prime == 11
    assert conditions["severe_collapse"].collapse_level == 0.5
