import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F

from src.data import DatasetConfig, generate_modular_arithmetic, get_all_conditions

def calculate_entropy(probs):
    """Calculate Shannon entropy for a probability distribution."""
    # Filter out 0s to avoid log(0)
    p = probs[probs > 0]
    return -np.sum(p * np.log(p))

def calculate_kl_divergence(p, q):
    """Calculate KL divergence KL(P || Q)."""
    # Filter out where p is 0, if p > 0 and q is 0 it's infinity
    mask = p > 0
    p = p[mask]
    q = q[mask]
    if np.any(q == 0):
        return float('inf')
    return np.sum(p * np.log(p / q))

def compute_data_metrics(output_file: Path):
    conditions = get_all_conditions()

    records = []

    prime = conditions['pure'].prime
    pure_probs = np.ones(prime) / prime

    for name, config in conditions.items():
        if name not in ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]:
            continue

        print(f"Generating data for {name}...")
        train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

        targets = train_tgt.numpy()
        counts = Counter(targets)

        empirical_probs = np.zeros(prime)
        for val, count in counts.items():
            empirical_probs[val] = count / len(targets)

        entropy = calculate_entropy(empirical_probs)
        kl_div = calculate_kl_divergence(empirical_probs, pure_probs)
        missing_targets = prime - len(counts)
        max_prob = np.max(empirical_probs)

        # Determine memorizability:
        # A simple proxy: examples that have target outputs which are very common
        # in the collapsed distribution are easier to "memorize" simply by learning
        # the marginal distribution (the collapse collapse).
        # We can calculate the expected predictability = sum_i p_i^2.
        memorizability_score = np.sum(empirical_probs ** 2)

        records.append({
            'condition': name,
            'collapse_level': config.collapse_level,
            'collapse_severity': config.collapse_severity,
            'entropy': entropy,
            'kl_divergence': kl_div,
            'missing_targets': missing_targets,
            'max_prob': max_prob,
            'memorizability_score': memorizability_score
        })

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Saved data metrics to {output_file}")

def main():
    out_dir = Path("analysis/data_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dataset_metrics.csv"
    compute_data_metrics(out_file)

if __name__ == "__main__":
    main()
