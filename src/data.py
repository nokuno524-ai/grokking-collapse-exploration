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
    seed: int = 42


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
