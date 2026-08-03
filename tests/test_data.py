import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data import (
    DatasetConfig,
    generate_modular_arithmetic,
    generate_group_multiplication,
    generate_binary_addition,
    generate_sparse_parity,
    generate_in_context_learning
)

def test_generate_modular_arithmetic():
    config = DatasetConfig(prime=7, train_fraction=0.5)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_pairs = 7 * 7
    expected_train = int(round(total_pairs * 0.5))
    expected_test = total_pairs - expected_train

    assert train_in.shape == (expected_train, 2)
    assert train_tgt.shape == (expected_train,)
    assert test_in.shape == (expected_test, 2)
    assert test_tgt.shape == (expected_test,)

    # Values should be within [0, p-1]
    assert torch.all(train_in >= 0) and torch.all(train_in < 7)
    assert torch.all(train_tgt >= 0) and torch.all(train_tgt < 7)

def test_multitask_datasets():
    config = DatasetConfig(prime=7, train_fraction=0.5)

    tasks = [
        generate_group_multiplication,
        generate_binary_addition,
        generate_sparse_parity,
        generate_in_context_learning
    ]

    for task_func in tasks:
        train_in, train_tgt, test_in, test_tgt = task_func(config)

        # Check shapes
        assert train_in.shape[1] == 2
        assert len(train_tgt.shape) == 1
        assert test_in.shape[1] == 2
        assert len(test_tgt.shape) == 1

        # We adjust effective prime in some datasets (e.g., binary addition -> next power of 2 divided by 2 -> 4)
        # So we just check that data is generated successfully.
        assert train_in.size(0) + test_in.size(0) > 0
