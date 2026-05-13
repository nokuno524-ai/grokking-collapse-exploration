"""
Multi-task data generation for grokking-collapse experiments.
Includes tasks for polynomial arithmetic, composition, permutation, and sorting.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Callable
from collections import Counter

from src.data import apply_collapse, DatasetConfig

def generate_polynomial_arithmetic(
    config: DatasetConfig, degree: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate polynomial arithmetic dataset: sum_{i=0}^{degree} a_i * x^i mod p.
    Inputs are (x, a_0, a_1, ..., a_degree).
    Targets are the evaluated polynomial mod p.
    """
    p = config.prime
    rng = np.random.RandomState(config.seed)

    # Generate random inputs
    # Let's generate a reasonable number of samples, say min(p**(degree+2), 10000)
    # Actually, let's just randomly sample to avoid memory issues for large degree/p
    num_samples = min(p ** (degree + 2), 10000)

    all_pairs = []
    all_targets = []

    # To ensure systematic generation if state space is small
    if p ** (degree + 2) <= 10000:
        import itertools
        ranges = [range(p)] * (degree + 2)
        for params in itertools.product(*ranges):
            x = params[0]
            coeffs = params[1:]

            y = 0
            for i, c in enumerate(coeffs):
                y = (y + c * (x ** i)) % p

            all_pairs.append(params)
            all_targets.append(y)
    else:
        for _ in range(num_samples):
            params = tuple(rng.randint(0, p, size=degree + 2))
            x = params[0]
            coeffs = params[1:]

            y = 0
            for i, c in enumerate(coeffs):
                y = (y + c * (x ** i)) % p

            all_pairs.append(params)
            all_targets.append(y)

    # Remove duplicates if randomly sampled
    if p ** (degree + 2) > 10000:
        unique_pairs = list(set(all_pairs))
        all_targets = []
        for params in unique_pairs:
            x = params[0]
            coeffs = params[1:]
            y = 0
            for i, c in enumerate(coeffs):
                y = (y + c * (x ** i)) % p
            all_targets.append(y)
        all_pairs = unique_pairs

    # Shuffle and split
    indices = rng.permutation(len(all_pairs))
    n_train = int(len(all_pairs) * config.train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_pairs = [all_pairs[i] for i in train_idx]
    train_targets_list = [all_targets[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_targets_list = [all_targets[i] for i in test_idx]

    # Apply collapse
    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, p,
            config.collapse_level, config.collapse_severity, rng
        )

    # Convert to tensors
    train_inputs = torch.tensor(train_pairs, dtype=torch.long)
    train_targets = torch.tensor(train_targets_list, dtype=torch.long)
    test_inputs = torch.tensor(test_pairs, dtype=torch.long)
    test_targets = torch.tensor(test_targets_list, dtype=torch.long)

    return train_inputs, train_targets, test_inputs, test_targets


def generate_composition_task(
    config: DatasetConfig, ops: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate composition task: f(g(h(x))) mod p, where f, g, h are operations.
    Inputs are (x, op1, op2, ..., op_k).
    Targets are the composed result mod p.
    Assume operations are addition by a constant, multiplication by a constant.
    To encode operations simply: op_idx = c + p * type (type 0: add, type 1: mult)
    Total operations = 2 * p.
    """
    p = config.prime
    rng = np.random.RandomState(config.seed)

    num_ops_types = 2 * p

    # Generate random samples
    num_samples = min(p * (num_ops_types ** ops), 10000)

    all_pairs = []
    all_targets = []

    if p * (num_ops_types ** ops) <= 10000:
        import itertools
        ranges = [range(p)] + [range(num_ops_types)] * ops
        for params in itertools.product(*ranges):
            x = params[0]
            val = x
            for op in params[1:]:
                op_type = op // p
                op_val = op % p
                if op_type == 0:
                    val = (val + op_val) % p
                else:
                    val = (val * op_val) % p

            all_pairs.append(params)
            all_targets.append(val)
    else:
        for _ in range(num_samples):
            x = rng.randint(0, p)
            operations = rng.randint(0, num_ops_types, size=ops)
            params = tuple([x] + list(operations))

            val = x
            for op in operations:
                op_type = op // p
                op_val = op % p
                if op_type == 0:
                    val = (val + op_val) % p
                else:
                    val = (val * op_val) % p

            all_pairs.append(params)
            all_targets.append(val)

        unique_pairs = list(set(all_pairs))
        all_targets = []
        for params in unique_pairs:
            x = params[0]
            val = x
            for op in params[1:]:
                op_type = op // p
                op_val = op % p
                if op_type == 0:
                    val = (val + op_val) % p
                else:
                    val = (val * op_val) % p
            all_targets.append(val)
        all_pairs = unique_pairs

    # Shuffle and split
    indices = rng.permutation(len(all_pairs))
    n_train = int(len(all_pairs) * config.train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_pairs = [all_pairs[i] for i in train_idx]
    train_targets_list = [all_targets[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_targets_list = [all_targets[i] for i in test_idx]

    # Apply collapse
    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, p,
            config.collapse_level, config.collapse_severity, rng
        )

    # Convert to tensors
    train_inputs = torch.tensor(train_pairs, dtype=torch.long)
    train_targets = torch.tensor(train_targets_list, dtype=torch.long)
    test_inputs = torch.tensor(test_pairs, dtype=torch.long)
    test_targets = torch.tensor(test_targets_list, dtype=torch.long)

    return train_inputs, train_targets, test_inputs, test_targets


def generate_permutation_task(
    config: DatasetConfig, n: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate permutation task: finding the inverse of a permutation.
    Inputs are (p_1, p_2, ..., p_n) representing a permutation of 0 to n-1.
    Targets are the inverse permutation (p^{-1}_1, ..., p^{-1}_n).
    Because our targets so far have been single integers, we'll redefine the target
    to be evaluating the inverse permutation at a given index.
    Inputs: (p_1, ..., p_n, idx)
    Target: The index j such that p_j = idx
    """
    rng = np.random.RandomState(config.seed)

    import itertools
    permutations = list(itertools.permutations(range(n)))

    all_pairs = []
    all_targets = []

    for perm in permutations:
        for idx in range(n):
            params = tuple(list(perm) + [idx])
            # Find j such that perm[j] == idx
            target = perm.index(idx)

            all_pairs.append(params)
            all_targets.append(target)

    # Shuffle and split
    indices = rng.permutation(len(all_pairs))
    n_train = int(len(all_pairs) * config.train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_pairs = [all_pairs[i] for i in train_idx]
    train_targets_list = [all_targets[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_targets_list = [all_targets[i] for i in test_idx]

    # Apply collapse
    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, n, # Domain of target is [0, n-1]
            config.collapse_level, config.collapse_severity, rng
        )

    # Convert to tensors
    train_inputs = torch.tensor(train_pairs, dtype=torch.long)
    train_targets = torch.tensor(train_targets_list, dtype=torch.long)
    test_inputs = torch.tensor(test_pairs, dtype=torch.long)
    test_targets = torch.tensor(test_targets_list, dtype=torch.long)

    return train_inputs, train_targets, test_inputs, test_targets


def generate_sorting_task(
    config: DatasetConfig, seq_len: int = 5, vocab_size: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate sorting task: predicting the sorted sequence.
    To fit single integer prediction, target is asking for the k-th smallest element.
    Inputs: (x_1, ..., x_seq_len, k)
    Target: The k-th smallest element of the sequence (0-indexed).
    """
    rng = np.random.RandomState(config.seed)

    num_samples = min(vocab_size ** seq_len * seq_len, 10000)

    all_pairs = []
    all_targets = []

    if vocab_size ** seq_len * seq_len <= 10000:
        import itertools
        ranges = [range(vocab_size)] * seq_len
        for seq in itertools.product(*ranges):
            sorted_seq = sorted(seq)
            for k in range(seq_len):
                params = tuple(list(seq) + [k])
                target = sorted_seq[k]

                all_pairs.append(params)
                all_targets.append(target)
    else:
        for _ in range(num_samples):
            seq = rng.randint(0, vocab_size, size=seq_len)
            k = rng.randint(0, seq_len)

            params = tuple(list(seq) + [k])
            sorted_seq = sorted(seq)
            target = sorted_seq[k]

            all_pairs.append(params)
            all_targets.append(target)

        unique_pairs = list(set(all_pairs))
        all_targets = []
        for params in unique_pairs:
            seq = params[:-1]
            k = params[-1]
            sorted_seq = sorted(seq)
            target = sorted_seq[k]
            all_targets.append(target)
        all_pairs = unique_pairs

    # Shuffle and split
    indices = rng.permutation(len(all_pairs))
    n_train = int(len(all_pairs) * config.train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_pairs = [all_pairs[i] for i in train_idx]
    train_targets_list = [all_targets[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_targets_list = [all_targets[i] for i in test_idx]

    # Apply collapse
    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, vocab_size, # Domain is [0, vocab_size-1]
            config.collapse_level, config.collapse_severity, rng
        )

    # Convert to tensors
    train_inputs = torch.tensor(train_pairs, dtype=torch.long)
    train_targets = torch.tensor(train_targets_list, dtype=torch.long)
    test_inputs = torch.tensor(test_pairs, dtype=torch.long)
    test_targets = torch.tensor(test_targets_list, dtype=torch.long)

    return train_inputs, train_targets, test_inputs, test_targets
