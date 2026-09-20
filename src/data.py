"""
Data generation for grokking-collapse experiments.
Generates modular arithmetic datasets with varying levels of synthetic data contamination
to simulate model collapse.
"""

import torch
import hashlib
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
    tree_depth: int = 1  # Number of recursive generation steps


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
            config.collapse_level, config.collapse_severity, config.seed, config.tree_depth
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
    collapse_level: float, collapse_severity: float, seed: int,
    tree_depth: int = 1
) -> Tuple[list, list]:
    """
    Simulate model collapse by replacing some targets with outputs from a "collapsed" model.
    """
    new_pairs = list(pairs)
    new_targets = list(targets)
    
    # We use round() instead of int() to prevent silent data truncation biases
    n_replace = int(round(len(targets) * collapse_level))
    if n_replace == 0:
        return new_pairs, new_targets

    for depth in range(tree_depth):
        # Derive generation-specific seed to ensure each generation is reproducible
        # We use a 32-bit compatible hash
        seed_str = f"{seed}_collapse_generation_{depth}"
        gen_seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(gen_seed)

        replace_idx = rng.choice(len(new_targets), n_replace, replace=False)

        # Compute target frequency distribution
        from collections import Counter
        target_counts = Counter(new_targets)
        total = len(new_targets)

        # Explicit sorting for cross-environment determinism
        sorted_keys = sorted(target_counts.keys())
        freq = {t: target_counts[t] / total for t in sorted_keys}

        # Create collapsed distribution: amplify common targets, suppress rare ones
        temp = max(0.1, 1.0 - collapse_severity)
        collapsed_probs = {}
        for t in range(prime):
            base_prob = freq.get(t, 1.0 / prime)
            collapsed_probs[t] = base_prob ** (1.0 / temp)

        # Normalize
        total_prob = sum(collapsed_probs.values())
        sorted_targets = sorted(collapsed_probs.keys())
        collapsed_probs = {t: collapsed_probs[t] / total_prob for t in sorted_targets}

        # Allocate exact integer counts per target to prevent distribution drift
        counts = {t: int(np.floor(collapsed_probs[t] * n_replace)) for t in sorted_targets}
        remainder = n_replace - sum(counts.values())
        if remainder > 0:
            fracs = {t: collapsed_probs[t] * n_replace - counts[t] for t in sorted_targets}
            sorted_by_frac = sorted(sorted_targets, key=lambda k: (-fracs[k], k))
            for k in sorted_by_frac[:remainder]:
                counts[k] += 1

        # Generate exact samples list and shuffle deterministically
        samples = []
        for t in sorted_targets:
            samples.extend([t] * counts[t])
        rng.shuffle(samples)

        for idx, new_target in zip(replace_idx, samples):
            new_targets[idx] = int(new_target)

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
