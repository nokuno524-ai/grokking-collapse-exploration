import pytest
import numpy as np
import torch
from src.data import generate_modular_arithmetic, DatasetConfig, get_all_conditions

def test_data_generation_shapes():
    config = DatasetConfig(prime=7, train_fraction=0.5, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_samples = 7 * 7
    expected_train = int(total_samples * 0.5)
    expected_test = total_samples - expected_train

    assert train_in.shape == (expected_train, 2)
    assert train_tgt.shape == (expected_train,)
    assert test_in.shape == (expected_test, 2)
    assert test_tgt.shape == (expected_test,)

def test_pure_data_correctness():
    config = DatasetConfig(prime=7, collapse_level=0.0)
    train_in, train_tgt, _, _ = generate_modular_arithmetic(config)

    for i in range(len(train_in)):
        a, b = train_in[i].tolist()
        expected = (a + b) % 7
        assert train_tgt[i].item() == expected

def test_collapse_distribution_shift():
    # Large prime to have enough samples for stats
    prime = 59
    # Pure data
    pure_config = DatasetConfig(prime=prime, collapse_level=0.0, seed=42)
    _, pure_tgt, _, _ = generate_modular_arithmetic(pure_config)

    # Severe collapse
    collapse_config = DatasetConfig(prime=prime, collapse_level=0.5, collapse_severity=0.9, seed=42)
    _, collapse_tgt, _, _ = generate_modular_arithmetic(collapse_config)

    # The targets should be different
    assert not torch.equal(pure_tgt, collapse_tgt)

    # In pure data, targets should be roughly uniformly distributed
    # In collapsed data, entropy should be lower (distribution shift)
    pure_counts = np.bincount(pure_tgt.numpy(), minlength=prime)
    collapse_counts = np.bincount(collapse_tgt.numpy(), minlength=prime)

    pure_probs = pure_counts / pure_counts.sum()
    collapse_probs = collapse_counts / collapse_counts.sum()

    pure_entropy = -np.sum(pure_probs[pure_probs > 0] * np.log(pure_probs[pure_probs > 0]))
    collapse_entropy = -np.sum(collapse_probs[collapse_probs > 0] * np.log(collapse_probs[collapse_probs > 0]))

    # Collapsed distribution should have lower entropy (more concentrated)
    assert collapse_entropy < pure_entropy
