import pytest
import torch
from src.data import DatasetConfig
from src.data_multi import (
    generate_polynomial_arithmetic,
    generate_composition_task,
    generate_permutation_task,
    generate_sorting_task
)

def test_generate_polynomial_arithmetic():
    config = DatasetConfig(prime=7, train_fraction=0.5, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_polynomial_arithmetic(config, degree=2)

    # 7^4 = 2401 total samples
    assert len(train_in) > 0
    assert len(test_in) > 0
    assert train_in.shape[1] == 4  # (x, a_0, a_1, a_2)
    assert train_tgt.shape == (len(train_in),)
    assert test_tgt.shape == (len(test_in),)

def test_generate_composition_task():
    config = DatasetConfig(prime=5, train_fraction=0.5, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_composition_task(config, ops=2)

    assert len(train_in) > 0
    assert len(test_in) > 0
    assert train_in.shape[1] == 3  # (x, op1, op2)
    assert train_tgt.shape == (len(train_in),)
    assert test_tgt.shape == (len(test_in),)

def test_generate_permutation_task():
    config = DatasetConfig(prime=7, train_fraction=0.5, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_permutation_task(config, n=4)

    # 4! * 4 = 96 total samples
    assert len(train_in) == 48
    assert len(test_in) == 48
    assert train_in.shape[1] == 5  # (p_1, ..., p_4, idx)
    assert train_tgt.shape == (len(train_in),)
    assert test_tgt.shape == (len(test_in),)

def test_generate_sorting_task():
    config = DatasetConfig(prime=7, train_fraction=0.5, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_sorting_task(config, seq_len=4, vocab_size=5)

    assert len(train_in) > 0
    assert len(test_in) > 0
    assert train_in.shape[1] == 5  # (x_1, ..., x_4, k)
    assert train_tgt.shape == (len(train_in),)
    assert test_tgt.shape == (len(test_in),)
