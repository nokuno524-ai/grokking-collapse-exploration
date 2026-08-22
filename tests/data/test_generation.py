import pytest
import torch
import numpy as np
from src.data.generation import DatasetConfig, generate_modular_arithmetic, apply_collapse, apply_label_noise

def test_dataset_generation_shapes():
    config = DatasetConfig(prime=59, train_fraction=0.3, seed=42)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_samples = 59 * 59
    n_train = int(total_samples * 0.3)
    n_test = total_samples - n_train

    assert train_in.shape == (n_train, 2)
    assert train_tgt.shape == (n_train,)
    assert test_in.shape == (n_test, 2)
    assert test_tgt.shape == (n_test,)

def test_determinism():
    config1 = DatasetConfig(prime=11, train_fraction=0.5, seed=42)
    train_in1, train_tgt1, test_in1, test_tgt1 = generate_modular_arithmetic(config1)

    config2 = DatasetConfig(prime=11, train_fraction=0.5, seed=42)
    train_in2, train_tgt2, test_in2, test_tgt2 = generate_modular_arithmetic(config2)

    assert torch.all(train_in1 == train_in2)
    assert torch.all(train_tgt1 == train_tgt2)

def test_apply_collapse_empty():
    rng = np.random.RandomState(42)
    new_pairs, new_targets = apply_collapse([], [], 59, 0.5, 0.5, rng)
    assert new_pairs == []
    assert new_targets == []

def test_apply_collapse_zero_level():
    rng = np.random.RandomState(42)
    pairs = [(1, 2), (3, 4)]
    targets = [3, 7]
    new_pairs, new_targets = apply_collapse(pairs, targets, 59, 0.0, 0.5, rng)
    assert new_pairs == pairs
    assert new_targets == targets

def test_apply_label_noise():
    rng = np.random.RandomState(42)
    pairs = [(1, 2)] * 100
    targets = [3] * 100
    new_pairs, new_targets = apply_label_noise(pairs, targets, 59, 1.0, rng)

    # All should be replaced
    assert new_targets != targets
    assert all(t != 3 for t in new_targets) # Noise replacement avoids the original target

def test_apply_label_noise_empty():
    rng = np.random.RandomState(42)
    new_pairs, new_targets = apply_label_noise([], [], 59, 1.0, rng)
    assert new_pairs == []
    assert new_targets == []
