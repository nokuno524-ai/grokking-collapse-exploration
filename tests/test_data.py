import torch
import pytest
from src.data import DatasetConfig, generate_modular_arithmetic

def test_data_generation():
    config = DatasetConfig(prime=7)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)
    assert train_in.shape[0] > 0

def test_data_determinism():
    config1 = DatasetConfig(prime=7, seed=42, base_seed=42, phase_seed=42, collapse_level=0.5, collapse_severity=0.5)
    train_in1, train_tgt1, _, _ = generate_modular_arithmetic(config1)

    config2 = DatasetConfig(prime=7, seed=42, base_seed=42, phase_seed=42, collapse_level=0.5, collapse_severity=0.5)
    train_in2, train_tgt2, _, _ = generate_modular_arithmetic(config2)

    assert torch.all(train_in1 == train_in2)
    assert torch.all(train_tgt1 == train_tgt2)

def test_data_different_phase_seed():
    config1 = DatasetConfig(prime=7, seed=42, base_seed=42, phase_seed=42, collapse_level=0.5, collapse_severity=0.5)
    train_in1, train_tgt1, _, _ = generate_modular_arithmetic(config1)

    config2 = DatasetConfig(prime=7, seed=42, base_seed=42, phase_seed=100, collapse_level=0.5, collapse_severity=0.5)
    train_in2, train_tgt2, _, _ = generate_modular_arithmetic(config2)

    # Input splits should be identical due to same base_seed
    assert torch.all(train_in1 == train_in2)
    # Target replacements should be different due to different phase_seed
    assert not torch.all(train_tgt1 == train_tgt2)
