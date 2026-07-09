import argparse
import json
import os
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.data import generate_modular_arithmetic, DatasetConfig, apply_collapse, apply_label_noise
from src.model import ModularArithmeticTransformer
from src.train import train, TrainConfig

def compute_kl_divergence(p_probs, q_probs):
    """Compute KL(P || Q) where P and Q are dicts mapping target to probability."""
    kl = 0.0
    for t, p_val in p_probs.items():
        if p_val > 0:
            q_val = q_probs.get(t, 1e-10)
            kl += p_val * math.log(p_val / q_val)
    return kl

def compute_information_content(probs, prime):
    """
    Compute effective information content in bits.
    Max entropy for uniform distribution over `prime` is log2(prime).
    Information content = Max Entropy - H(probs)
    """
    entropy = 0.0
    for p_val in probs.values():
        if p_val > 0:
            entropy -= p_val * math.log2(p_val)
    max_entropy = math.log2(prime)
    return max_entropy - entropy

def calculate_empirical_distribution(targets, prime):
    from collections import Counter
    counts = Counter(targets)
    total = len(targets)
    return {t: counts.get(t, 0) / total for t in range(prime)}

def analyze_distributions(prime, train_targets_pure, train_targets_corrupted):
    pure_probs = calculate_empirical_distribution(train_targets_pure.tolist(), prime)
    corrupted_probs = calculate_empirical_distribution(train_targets_corrupted.tolist(), prime)

    kl_div = compute_kl_divergence(pure_probs, corrupted_probs)
    info_content = compute_information_content(corrupted_probs, prime)

    return kl_div, info_content

def search_matched_noise_rate(seed, target_acc, tolerance=0.05, max_attempts=5):
    """
    Find equivalent noise rate that achieves target final test accuracy
    by doing a small linear search or a pre-defined set of rates.
    For simplicity and stability, we test a few standard noise rates and find the closest match.
    """
    candidate_rates = [0.05, 0.10, 0.15, 0.20, 0.30]
    best_rate = candidate_rates[0]
    best_diff = float('inf')

    print(f"Searching for matched noise rate for target acc {target_acc:.3f}...")

    # We will do a short train run (e.g. 5000 steps) to estimate final accuracy
    # to find the matched noise rate.
    for rate in candidate_rates:
        noise_train_config = TrainConfig(
            prime=59,
            train_fraction=0.3,
            collapse_level=0.0,
            noise_fraction=rate,
            seed=seed,
            output_dir=f"results/tmp_noise_search_s{seed}_r{rate}",
            condition_name="noise_search",
            max_steps=5000, # fast test
            save_every=5000,
            eval_every=1000
        )
        state = train(noise_train_config)
        diff = abs(state.test_acc - target_acc)
        print(f"Candidate noise {rate}: acc {state.test_acc:.3f} (diff {diff:.3f})")
        if diff < best_diff:
            best_diff = diff
            best_rate = rate
        if diff < tolerance:
            break

    return best_rate

def main():
    parser = argparse.ArgumentParser(description="Collapse vs Noise Experiment")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results/phase2_collapse_vs_noise")
    parser.add_argument("--max_steps", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    collapse_levels = {
        "low_collapse": 0.05,
        "medium_collapse": 0.15,
        "severe_collapse": 0.50
    }

    for c_name, c_level in collapse_levels.items():
        results[c_name] = {"collapse": [], "matched_noise": []}

        for seed in range(42, 42 + args.num_seeds):
            print(f"\n--- Condition: {c_name} (level={c_level}), Seed {seed} ---")

            # Pure data for baseline distribution
            pure_config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0, noise_fraction=0.0, seed=seed)
            _, train_tgt_pure, _, _ = generate_modular_arithmetic(pure_config)

            # Collapse Data
            collapse_config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=c_level, collapse_severity=0.5, seed=seed)
            _, train_tgt_collapse, _, _ = generate_modular_arithmetic(collapse_config)

            kl_collapse, info_collapse = analyze_distributions(59, train_tgt_pure, train_tgt_collapse)
            print(f"Collapse KL: {kl_collapse:.4f}, Info Content: {info_collapse:.4f} bits")

            # Train Collapse Model
            collapse_train_config = TrainConfig(
                prime=59,
                train_fraction=0.3,
                collapse_level=c_level,
                collapse_severity=0.5,
                noise_fraction=0.0,
                seed=seed,
                output_dir=os.path.join(args.output_dir, f"{c_name}_collapse_s{seed}"),
                condition_name="collapse",
                max_steps=args.max_steps,
                save_every=100,
            )

            print("Training Collapse Model...")
            collapse_state = train(collapse_train_config)

            results[c_name]["collapse"].append({
                "seed": seed,
                "kl_divergence": kl_collapse,
                "info_content": info_collapse,
                "grokking_step": collapse_state.grokking_step,
                "final_weight_norm": collapse_state.weight_norm,
                "final_test_acc": collapse_state.test_acc,
                "history": collapse_state.history
            })

            # Search for equivalent noise rate based on collapse final test acc
            matched_noise_rate = search_matched_noise_rate(seed, collapse_state.test_acc)
            print(f"Matched noise rate found: {matched_noise_rate}")

            # Noise Data
            noise_config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0, noise_fraction=matched_noise_rate, seed=seed)
            _, train_tgt_noise, _, _ = generate_modular_arithmetic(noise_config)

            kl_noise, info_noise = analyze_distributions(59, train_tgt_pure, train_tgt_noise)
            print(f"Noise KL: {kl_noise:.4f}, Info Content: {info_noise:.4f} bits")

            noise_train_config = TrainConfig(
                prime=59,
                train_fraction=0.3,
                collapse_level=0.0,
                noise_fraction=matched_noise_rate,
                seed=seed,
                output_dir=os.path.join(args.output_dir, f"{c_name}_noise_s{seed}"),
                condition_name="noise",
                max_steps=args.max_steps,
                save_every=100,
            )

            print("Training Matched Noise Model...")
            noise_state = train(noise_train_config)

            results[c_name]["matched_noise"].append({
                "seed": seed,
                "matched_noise_rate": matched_noise_rate,
                "kl_divergence": kl_noise,
                "info_content": info_noise,
                "grokking_step": noise_state.grokking_step,
                "final_weight_norm": noise_state.weight_norm,
                "final_test_acc": noise_state.test_acc,
                "history": noise_state.history
            })

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Experiment completed successfully.")

if __name__ == "__main__":
    main()
