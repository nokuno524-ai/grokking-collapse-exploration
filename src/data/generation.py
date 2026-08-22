import torch
import numpy as np
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

@dataclass
class DatasetConfig:
    prime: int = 59
    train_fraction: float = 0.3
    collapse_level: float = 0.0
    collapse_severity: float = 0.5
    noise_fraction: float = 0.0
    seed: int = 42

def hash_config(config: DatasetConfig) -> str:
    config_dict = asdict(config)
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def apply_collapse(pairs: list, targets: list, prime: int,
    collapse_level: float, collapse_severity: float, rng: np.random.RandomState) -> Tuple[list, list]:
    if collapse_level == 0.0:
        return list(pairs), list(targets)
    if len(targets) == 0:
        return list(pairs), list(targets)
    
    n_replace = int(len(targets) * collapse_level)
    replace_idx = rng.choice(len(targets), n_replace, replace=False)
    
    from collections import Counter
    target_counts = Counter(targets)
    total = len(targets)
    freq = {t: c / total for t, c in target_counts.items()}
    
    temp = max(0.1, 1.0 - collapse_severity)
    collapsed_probs = {}
    for t in range(prime):
        base_prob = freq.get(t, 1.0 / prime)
        collapsed_probs[t] = base_prob ** (1.0 / temp)
    
    total_prob = sum(collapsed_probs.values())
    if total_prob == 0:
        collapsed_probs = {t: 1.0 / prime for t in range(prime)}
    else:
        collapsed_probs = {t: p / total_prob for t, p in collapsed_probs.items()}
    
    collapsed_targets = list(collapsed_probs.keys())
    collapsed_weights = [collapsed_probs[t] for t in collapsed_targets]
    
    new_pairs = list(pairs)
    new_targets = list(targets)
    
    for idx in replace_idx:
        new_target = rng.choice(collapsed_targets, p=collapsed_weights)
        new_targets[idx] = int(new_target)
    
    return new_pairs, new_targets

def apply_label_noise(pairs: list, targets: list, prime: int,
    noise_fraction: float, rng: np.random.RandomState) -> Tuple[list, list]:
    n_replace = int(len(targets) * noise_fraction)
    if n_replace == 0 or len(targets) == 0:
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

def generate_modular_arithmetic(config: DatasetConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p = config.prime
    rng = np.random.RandomState(config.seed)

    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    all_targets = [(a + b) % p for a, b in all_pairs]

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

    train_inputs = torch.tensor(train_pairs, dtype=torch.long)
    train_targets = torch.tensor(train_targets_list, dtype=torch.long)
    test_inputs = torch.tensor(test_pairs, dtype=torch.long)
    test_targets = torch.tensor(test_targets_list, dtype=torch.long)

    return train_inputs, train_targets, test_inputs, test_targets

def save_dataset(
    out_dir: Path,
    gen_idx: int,
    config: DatasetConfig,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    parent_gen: Optional[int] = None
):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generation": gen_idx,
        "parent_generation": parent_gen,
        "config": asdict(config),
        "config_hash": hash_config(config)
    }
    manifest_path = out_dir / f"manifest_gen{gen_idx}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    dataset = {
        "train_inputs": train_inputs.tolist(),
        "train_targets": train_targets.tolist(),
        "test_inputs": test_inputs.tolist(),
        "test_targets": test_targets.tolist(),
    }
    dataset_path = out_dir / f"dataset_gen{gen_idx}.jsonl"
    with open(dataset_path, "w") as f:
        f.write(json.dumps(dataset) + "\n")
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
