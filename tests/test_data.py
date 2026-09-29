import pytest
import torch
from src.data import DatasetConfig, generate_modular_arithmetic

def test_data_generation_invalid_prime():
    config = DatasetConfig(prime=1)
    with pytest.raises(ValueError, match="Prime modulus must be > 1"):
        generate_modular_arithmetic(config)

    config = DatasetConfig(prime=0)
    with pytest.raises(ValueError, match="Prime modulus must be > 1"):
        generate_modular_arithmetic(config)

def test_data_generation_valid():
    config = DatasetConfig(prime=5)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    assert isinstance(train_in, torch.Tensor)
    assert isinstance(train_tgt, torch.Tensor)
    assert train_in.shape[1] == 2
