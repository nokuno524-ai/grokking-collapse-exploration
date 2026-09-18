"""
Data generation for grokking-collapse experiments.
Generates modular arithmetic datasets with varying levels of synthetic data contamination
to simulate model collapse.
"""

import torch
import numpy as np
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    prime: int = 59  # Modular arithmetic modulus
    train_fraction: float = 0.3  # Fraction of data for training
    collapse_level: float = 0.0  # Fraction of training data replaced by synthetic
    collapse_severity: float = 0.5  # How much the synthetic generator has "collapsed" (0=fresh, 1=fully collapsed)
    noise_fraction: float = 0.0  # Fraction of training labels replaced with uniform random labels (baseline)
    seed: int = 42
    tree_depth: int = 1  # Number of recursive generations simulating collapse


def get_condition_seed(base_seed: int, condition_name: str) -> int:
    """Hash the condition name into the base seed to prevent seed reuse across conditions."""
    h = hashlib.sha256(f"{base_seed}_{condition_name}".encode()).hexdigest()
    return int(h, 16) % (2**32)


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
            config.collapse_level, config.collapse_severity, rng,
            config.tree_depth
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
    pairs: List[Tuple[int, int]], targets: List[int], prime: int,
    collapse_level: float, collapse_severity: float, rng: np.random.RandomState,
    tree_depth: int = 1
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Simulate model collapse by replacing some targets with outputs from a "collapsed" model.
    """
    n_replace = int(len(targets) * collapse_level)
    if n_replace == 0:
        return list(pairs), list(targets)

    replace_idx = rng.choice(len(targets), n_replace, replace=False)
    
    # Compute target frequency distribution exactly
    target_counts = Counter(targets)
    total = len(targets)
    # Ensure all targets are present and strictly sorted by keys to guarantee cross-environment determinism
    current_probs = {t: target_counts.get(t, 0) / total for t in range(prime)}
    
    # Create collapsed distribution: amplify common targets, suppress rare ones
    # Use temperature to control severity
    temp = max(0.1, 1.0 - collapse_severity)
    
    # Recursive generations simulating model collapse
    for _ in range(tree_depth):
        next_probs = {}
        for t in range(prime):
            # Apply temperature to the probabilities
            next_probs[t] = current_probs[t] ** (1.0 / temp) if current_probs[t] > 0 else 0.0

        # Normalize
        total_prob = sum(next_probs.values())
        if total_prob > 0:
            current_probs = {t: p / total_prob for t, p in next_probs.items()}
        else:
            # Fallback to uniform if all probabilities somehow vanished
            current_probs = {t: 1.0 / prime for t in range(prime)}
    
    collapsed_targets = sorted(list(current_probs.keys()))
    collapsed_weights = [current_probs[t] for t in collapsed_targets]

    # Exact ratio attainment for the sampler to avoid binomial distribution drift
    # Distribute n_replace items among categories according to collapsed_weights using largest remainder method
    expected_counts = [collapsed_weights[t] * n_replace for t in range(prime)]
    integer_counts = [int(c) for c in expected_counts]
    remainders = [(c - int(c), t) for t, c in enumerate(expected_counts)]

    missing = n_replace - sum(integer_counts)
    remainders.sort(key=lambda x: x[0], reverse=True)

    for i in range(missing):
        integer_counts[remainders[i][1]] += 1

    synthetic_targets_to_inject = []
    for t in range(prime):
        synthetic_targets_to_inject.extend([t] * integer_counts[t])

    # Shuffle the injected targets securely
    rng.shuffle(synthetic_targets_to_inject)
    
    new_pairs = list(pairs)
    new_targets = list(targets)
    
    for i, idx in enumerate(replace_idx):
        new_targets[idx] = synthetic_targets_to_inject[i]
    
    return new_pairs, new_targets


def apply_label_noise(
    pairs: List[Tuple[int, int]], targets: List[int], prime: int,
    noise_fraction: float, rng: np.random.RandomState,
) -> Tuple[List[Tuple[int, int]], List[int]]:
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


def get_all_conditions(prime: int = 59, seed: int = 42) -> Dict[str, DatasetConfig]:
    """Get all experimental conditions."""
    return {
        "pure": DatasetConfig(prime=prime, collapse_level=0.0, seed=get_condition_seed(seed, "pure")),
        "low_collapse": DatasetConfig(prime=prime, collapse_level=0.05, collapse_severity=0.3, seed=get_condition_seed(seed, "low_collapse")),
        "medium_collapse": DatasetConfig(prime=prime, collapse_level=0.15, collapse_severity=0.5, seed=get_condition_seed(seed, "medium_collapse")),
        "high_collapse": DatasetConfig(prime=prime, collapse_level=0.30, collapse_severity=0.7, seed=get_condition_seed(seed, "high_collapse")),
        "severe_collapse": DatasetConfig(prime=prime, collapse_level=0.50, collapse_severity=0.9, seed=get_condition_seed(seed, "severe_collapse")),
    }


if __name__ == "__main__":
    # Quick test
    conditions = get_all_conditions()
    for name, config in conditions.items():
        train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)
        print(f"{name}: train={train_in.shape}, test={test_in.shape}, "
              f"unique_targets={len(set(train_tgt.tolist()))}")
