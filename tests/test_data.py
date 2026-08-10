import pytest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import DatasetConfig, generate_modular_arithmetic, get_all_conditions

def test_data_generation_shapes():
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    # Total pairs = 59 * 59 = 3481
    total = 59 * 59
    n_train = int(round(total * 0.3))
    n_test = total - n_train

    assert train_in.shape == (n_train, 2)
    assert train_tgt.shape == (n_train,)
    assert test_in.shape == (n_test, 2)
    assert test_tgt.shape == (n_test,)

def test_no_data_leakage():
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0)
    train_in, _, test_in, _ = generate_modular_arithmetic(config)

    # Convert rows to tuples for intersection checking
    train_set = set(tuple(x.tolist()) for x in train_in)
    test_set = set(tuple(x.tolist()) for x in test_in)

    # Ensure no overlap
    assert len(train_set.intersection(test_set)) == 0

def test_collapse_level():
    # Pure condition
    config_pure = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0)
    _, train_tgt_pure, _, _ = generate_modular_arithmetic(config_pure)
    unique_pure = len(torch.unique(train_tgt_pure))

    # Severe collapse
    config_collapse = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.5, collapse_severity=0.9)
    _, train_tgt_collapse, _, _ = generate_modular_arithmetic(config_collapse)
    unique_collapse = len(torch.unique(train_tgt_collapse))

    # Collapse should generally reduce the number of unique targets generated,
    # or alter the distribution significantly.
    assert unique_collapse <= unique_pure

def test_get_all_conditions():
    conditions = get_all_conditions()
    assert "pure" in conditions
    assert "severe_collapse" in conditions
    assert conditions["pure"].collapse_level == 0.0
    assert conditions["severe_collapse"].collapse_level > 0.0
