"""
Data generation for grokking-collapse experiments.
Generates modular arithmetic datasets with varying levels of synthetic data contamination
to simulate model collapse.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    prime: int = 59  # Modular arithmetic modulus
    train_fraction: float = 0.3  # Fraction of data for training
    collapse_level: float = 0.0  # Fraction of training data replaced by synthetic
    collapse_severity: float = 0.5  # How much the synthetic generator has "collapsed" (0=fresh, 1=fully collapsed)
    noise_fraction: float = 0.0  # Fraction of training labels replaced with uniform random labels (baseline)
    seed: int = 42


def generate_polynomial_identity(config: DatasetConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a polynomial identity dataset: (x, y) -> (x^2 + xy + y^2) mod p.
    """
    p = config.prime
    rng = np.random.RandomState(config.seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    all_targets = [(a*a + a*b + b*b) % p for a, b in all_pairs]

    indices = rng.permutation(len(all_pairs))
    n_train = int(len(all_pairs) * config.train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_pairs = [all_pairs[i] for i in train_idx]
    train_targets_list = [all_targets[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_targets_list = [all_targets[i] for i in test_idx]

    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, p,
            config.collapse_level, config.collapse_severity, rng
        )
    if config.noise_fraction > 0:
        train_pairs, train_targets_list = apply_label_noise(
            train_pairs, train_targets_list, p,
            config.noise_fraction, rng,
        )

    return (torch.tensor(train_pairs, dtype=torch.long),
            torch.tensor(train_targets_list, dtype=torch.long),
            torch.tensor(test_pairs, dtype=torch.long),
            torch.tensor(test_targets_list, dtype=torch.long))

def generate_sparse_parity(config: DatasetConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate sparse parity dataset. Input is sequence of length 2, values in [0, p).
    Target is (x + y) % 2.
    """
    p = config.prime
    rng = np.random.RandomState(config.seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    all_targets = [(a + b) % 2 for a, b in all_pairs]

    indices = rng.permutation(len(all_pairs))
    n_train = int(len(all_pairs) * config.train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_pairs = [all_pairs[i] for i in train_idx]
    train_targets_list = [all_targets[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_targets_list = [all_targets[i] for i in test_idx]

    # For parity, output is binary, so collapse is just shifting to all 0s or all 1s.
    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, 2, # Prime is 2 for output space
            config.collapse_level, config.collapse_severity, rng
        )
    if config.noise_fraction > 0:
        train_pairs, train_targets_list = apply_label_noise(
            train_pairs, train_targets_list, 2,
            config.noise_fraction, rng,
        )

    return (torch.tensor(train_pairs, dtype=torch.long),
            torch.tensor(train_targets_list, dtype=torch.long),
            torch.tensor(test_pairs, dtype=torch.long),
            torch.tensor(test_targets_list, dtype=torch.long))

def generate_digit_sorting(config: DatasetConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate digit sorting dataset. Task is to predict whether a > b.
    Target is 1 if a > b else 0.
    """
    p = config.prime
    rng = np.random.RandomState(config.seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    all_targets = [1 if a > b else 0 for a, b in all_pairs]

    indices = rng.permutation(len(all_pairs))
    n_train = int(len(all_pairs) * config.train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_pairs = [all_pairs[i] for i in train_idx]
    train_targets_list = [all_targets[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_targets_list = [all_targets[i] for i in test_idx]

    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, 2, # Binary output
            config.collapse_level, config.collapse_severity, rng
        )
    if config.noise_fraction > 0:
        train_pairs, train_targets_list = apply_label_noise(
            train_pairs, train_targets_list, 2,
            config.noise_fraction, rng,
        )

    return (torch.tensor(train_pairs, dtype=torch.long),
            torch.tensor(train_targets_list, dtype=torch.long),
            torch.tensor(test_pairs, dtype=torch.long),
            torch.tensor(test_targets_list, dtype=torch.long))


def get_task_loaders(task_name: str, config: DatasetConfig, batch_size: int = 512) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict]:
    """
    Task registry that returns dataloaders and metadata for standard grokking tasks.

    Args:
        task_name: 'modular_arithmetic', 'polynomial_identity', 'sparse_parity', 'digit_sorting'
        config: DatasetConfig
        batch_size: Batch size for dataloaders
    """
    task_map = {
        "modular_arithmetic": generate_modular_arithmetic,
        "polynomial_identity": generate_polynomial_identity,
        "sparse_parity": generate_sparse_parity,
        "digit_sorting": generate_digit_sorting,
    }

    if task_name not in task_map:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(task_map.keys())}")

    generate_fn = task_map[task_name]
    train_in, train_tgt, test_in, test_tgt = generate_fn(config)

    train_dataset = torch.utils.data.TensorDataset(train_in, train_tgt)
    test_dataset = torch.utils.data.TensorDataset(test_in, test_tgt)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=loader_generator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    output_size_map = {
        "modular_arithmetic": config.prime,
        "polynomial_identity": config.prime,
        "sparse_parity": 2,
        "digit_sorting": 2,
    }

    metadata = {
        "task": task_name,
        "vocab_size": config.prime,
        "output_classes": output_size_map[task_name],
        "train_size": len(train_dataset),
        "test_size": len(test_dataset)
    }

    return train_loader, test_loader, metadata


def generate_modular_arithmetic(config: DatasetConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate (a, b) -> (a + b) mod p dataset.
    
    Returns:
        train_inputs, train_targets, test_inputs, test_targets
        Inputs are (a, b) pairs encoded as token indices.
        Targets are (a + b) mod p.
    """
    p = config.prime
    rng = np.random.RandomState(config.seed)
    
    # Generate all possible (a, b) pairs
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    all_targets = [(a + b) % p for a, b in all_pairs]
    
    # Shuffle and split
    indices = rng.permutation(len(all_pairs))
    n_train = int(len(all_pairs) * config.train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_pairs = [all_pairs[i] for i in train_idx]
    train_targets_list = [all_targets[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_targets_list = [all_targets[i] for i in test_idx]
    
    # Apply collapse: replace some training examples with "collapsed" outputs
    if config.collapse_level > 0:
        train_pairs, train_targets_list = apply_collapse(
            train_pairs, train_targets_list, p,
            config.collapse_level, config.collapse_severity, rng
        )

    # Apply uniform random label noise (baseline ablation, mutually independent of collapse)
    if config.noise_fraction > 0:
        train_pairs, train_targets_list = apply_label_noise(
            train_pairs, train_targets_list, p,
            config.noise_fraction, rng,
        )

    # Convert to tensors
    train_inputs = torch.tensor(train_pairs, dtype=torch.long)
    train_targets = torch.tensor(train_targets_list, dtype=torch.long)
    test_inputs = torch.tensor(test_pairs, dtype=torch.long)
    test_targets = torch.tensor(test_targets_list, dtype=torch.long)
    
    return train_inputs, train_targets, test_inputs, test_targets


def apply_collapse(
    pairs: list, targets: list, prime: int,
    collapse_level: float, collapse_severity: float, rng: np.random.RandomState
) -> Tuple[list, list]:
    """
    Simulate model collapse by replacing some targets with outputs from a "collapsed" model.
    
    A collapsed model has:
    - Narrowed output distribution (favors common results)
    - Occasional errors (assigns probability mass incorrectly)
    - Loss of rare outputs
    """
    n_replace = int(len(targets) * collapse_level)
    replace_idx = rng.choice(len(targets), n_replace, replace=False)
    
    # Compute target frequency distribution
    from collections import Counter
    target_counts = Counter(targets)
    total = len(targets)
    freq = {t: c / total for t, c in target_counts.items()}
    
    # Create collapsed distribution: amplify common targets, suppress rare ones
    # Use temperature to control severity
    temp = max(0.1, 1.0 - collapse_severity)
    collapsed_probs = {}
    for t in range(prime):
        base_prob = freq.get(t, 1.0 / prime)
        collapsed_probs[t] = base_prob ** (1.0 / temp)
    
    # Normalize
    total_prob = sum(collapsed_probs.values())
    collapsed_probs = {t: p / total_prob for t, p in collapsed_probs.items()}
    
    # Sample from collapsed distribution
    collapsed_targets = list(collapsed_probs.keys())
    collapsed_weights = [collapsed_probs[t] for t in collapsed_targets]
    
    new_pairs = list(pairs)
    new_targets = list(targets)
    
    for idx in replace_idx:
        # Replace target with sample from collapsed distribution
        new_target = rng.choice(collapsed_targets, p=collapsed_weights)
        new_targets[idx] = int(new_target)
        # Optionally also corrupt the pair (simulating input collapse)
        # For now, keep inputs clean — only corrupt outputs
    
    return new_pairs, new_targets


def apply_label_noise(
    pairs: list, targets: list, prime: int,
    noise_fraction: float, rng: np.random.RandomState,
) -> Tuple[list, list]:
    """
    Replace a fraction of training labels with uniform random labels in [0, prime).
    The new label is drawn uniformly from the (prime-1) values different from the original
    so the corruption is always observable.
    """
    n_replace = int(len(targets) * noise_fraction)
    if n_replace == 0:
        return list(pairs), list(targets)
    replace_idx = rng.choice(len(targets), n_replace, replace=False)

    new_pairs = list(pairs)
    new_targets = list(targets)
    for idx in replace_idx:
        original = new_targets[idx]
        candidate = int(rng.randint(0, prime - 1))
        if candidate >= original:
            candidate += 1
        new_targets[idx] = candidate
    return new_pairs, new_targets


def get_all_conditions(prime: int = 59, seed: int = 42) -> dict:
    """Get all experimental conditions."""
    return {
        "pure": DatasetConfig(prime=prime, collapse_level=0.0, seed=seed),
        "low_collapse": DatasetConfig(prime=prime, collapse_level=0.05, collapse_severity=0.3, seed=seed),
        "medium_collapse": DatasetConfig(prime=prime, collapse_level=0.15, collapse_severity=0.5, seed=seed),
        "high_collapse": DatasetConfig(prime=prime, collapse_level=0.30, collapse_severity=0.7, seed=seed),
        "severe_collapse": DatasetConfig(prime=prime, collapse_level=0.50, collapse_severity=0.9, seed=seed),
    }


if __name__ == "__main__":
    # Quick test
    conditions = get_all_conditions()
    for name, config in conditions.items():
        train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)
        print(f"{name}: train={train_in.shape}, test={test_in.shape}, "
              f"unique_targets={len(set(train_tgt.tolist()))}")
