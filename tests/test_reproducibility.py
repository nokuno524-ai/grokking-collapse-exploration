import pytest
import torch
import numpy as np
from src.data import generate_modular_arithmetic, DatasetConfig

def test_data_generation_reproducibility():
    """Verify that using the same seed produces identical data."""
    config1 = DatasetConfig(prime=59, collapse_level=0.5, collapse_severity=0.8, seed=42)
    train_in1, train_tgt1, test_in1, test_tgt1 = generate_modular_arithmetic(config1)

    config2 = DatasetConfig(prime=59, collapse_level=0.5, collapse_severity=0.8, seed=42)
    train_in2, train_tgt2, test_in2, test_tgt2 = generate_modular_arithmetic(config2)

    assert torch.equal(train_in1, train_in2)
    assert torch.equal(train_tgt1, train_tgt2)
    assert torch.equal(test_in1, test_in2)
    assert torch.equal(test_tgt1, test_tgt2)

def test_different_seeds_produce_different_data():
    """Verify that different seeds produce different splits/corruption."""
    config1 = DatasetConfig(prime=59, collapse_level=0.5, collapse_severity=0.8, seed=42)
    train_in1, train_tgt1, _, _ = generate_modular_arithmetic(config1)

    config2 = DatasetConfig(prime=59, collapse_level=0.5, collapse_severity=0.8, seed=43)
    train_in2, train_tgt2, _, _ = generate_modular_arithmetic(config2)

    # Should not be identical
    assert not torch.equal(train_in1, train_in2)
