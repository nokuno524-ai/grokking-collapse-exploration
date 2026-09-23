"""
Script to compare grokking cliff timings across different training regimes.
Iterates over results directories, detects the grokking step for each run,
and performs statistical hypothesis testing.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from src.analysis.grok_detector.detectors import (
    plateau_detector,
    max_2nd_derivative_detector,
    fit_sigmoid_transition
)
from src.analysis.grok_detector.stats import (
    mann_whitney_u_test,
    permutation_test,
    hedges_g,
    bootstrap_ci,
    holm_correction
)

def load_run_history(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load steps and test accuracy from a run's results.json."""
    results_file = run_dir / "results.json"
    if not results_file.exists():
        return np.array([]), np.array([])

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        history = data.get("history", [])
        if not history:
            return np.array([]), np.array([])

        steps = np.array([h["step"] for h in history])
        accs = np.array([h.get("test_acc", h.get("accuracy", 0.0)) for h in history])

        return steps, accs
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return np.array([]), np.array([])

def detect_cliffs(results_dir: Path, method: str = 'plateau', threshold: float = 0.95) -> Dict[str, List[float]]:
    """
    Extract grokking steps for all runs in the specified directory.
    Assumes structure: results_dir / condition_name / run_name / results.json
    """
    condition_cliffs = {}

    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist.")
        return condition_cliffs

    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir():
            continue

        cliffs = []
        for run_dir in condition_dir.iterdir():
            if not run_dir.is_dir():
                continue

            steps, accs = load_run_history(run_dir)
            if len(steps) == 0:
                continue

            if method == 'plateau':
                cliff = plateau_detector(steps, accs, threshold=threshold)
            elif method == 'derivative':
                cliff = max_2nd_derivative_detector(steps, accs)
            elif method == 'sigmoid':
                cliff, _ = fit_sigmoid_transition(steps, accs)
            else:
                raise ValueError(f"Unknown detection method: {method}")

            if cliff is not None:
                cliffs.append(float(cliff))
            else:
                cliffs.append(float('nan'))

        if cliffs:
            condition_cliffs[condition_dir.name] = cliffs

    return condition_cliffs

def print_summary_stats(condition_cliffs: Dict[str, List[float]]):
    """Print summary statistics for each condition."""
    print(f"{'Condition':<20} | {'N (Grokked)':<12} | {'Mean Step':<10} | {'95% CI':<20}")
    print("-" * 68)

    for condition, cliffs in sorted(condition_cliffs.items()):
        valid_cliffs = [c for c in cliffs if not np.isnan(c)]
        n_total = len(cliffs)
        n_grokked = len(valid_cliffs)

        if n_grokked > 0:
            mean = np.mean(valid_cliffs)
            ci_low, ci_high = bootstrap_ci(valid_cliffs)
            ci_str = f"[{ci_low:.0f}, {ci_high:.0f}]"
            print(f"{condition:<20} | {n_grokked}/{n_total:<11} | {mean:<10.1f} | {ci_str:<20}")
        else:
            print(f"{condition:<20} | {n_grokked}/{n_total:<11} | {'N/A':<10} | {'N/A':<20}")

def run_pairwise_comparisons(condition_cliffs: Dict[str, List[float]], base_condition: str = None):
    """Run statistical comparisons between conditions."""
    conditions = sorted(list(condition_cliffs.keys()))
    if len(conditions) < 2:
        print("Not enough conditions for pairwise comparison.")
        return

    if base_condition is not None and base_condition not in condition_cliffs:
        print(f"Base condition '{base_condition}' not found.")
        return

    # If no base condition specified, use the first one (often "pure" or "clean")
    if base_condition is None:
        # Try to find "pure" or something similar
        for c in conditions:
            if "pure" in c.lower() or "clean" in c.lower():
                base_condition = c
                break
        if base_condition is None:
            base_condition = conditions[0]

    print(f"\nStatistical Comparisons (Base: {base_condition})")
    header_g = "Hedge's g"
    print(f"{'Condition':<20} | {'Mann-Whitney p':<15} | {'Holm-Corr p':<15} | {header_g:<10}")
    print("-" * 68)

    base_data = condition_cliffs[base_condition]

    comparisons = []
    p_values = []

    for condition in conditions:
        if condition == base_condition:
            continue

        test_data = condition_cliffs[condition]

        # Mann-Whitney U test
        _, p_val = mann_whitney_u_test(base_data, test_data)

        # Hedge's g effect size
        effect_size = hedges_g(test_data, base_data) # Positive means test group groks LATER

        comparisons.append((condition, p_val, effect_size))
        p_values.append(p_val)

    # Apply multiple comparison correction
    corrected_p_values = holm_correction(p_values)

    for (condition, orig_p, effect_size), corr_p in zip(comparisons, corrected_p_values):
        if np.isnan(orig_p):
            p_str, corr_p_str, eff_str = "N/A", "N/A", "N/A"
        else:
            p_str = f"{orig_p:.4f}"
            corr_p_str = f"{corr_p:.4f}"
            eff_str = f"{effect_size:.2f}"

        print(f"{condition:<20} | {p_str:<15} | {corr_p_str:<15} | {eff_str:<10}")


def main():
    parser = argparse.ArgumentParser(description="Compare grokking cliff timings across regimes.")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing subdirectories for each condition (which in turn contain run directories).")
    parser.add_argument("--method", type=str, choices=['plateau', 'derivative', 'sigmoid'], default='plateau',
                        help="Method for detecting the grokking changepoint.")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Threshold for plateau detector (ignored for other methods).")
    parser.add_argument("--base-condition", type=str, default=None,
                        help="The condition to compare all others against.")

    args = parser.parse_args()

    print(f"Detecting cliffs using '{args.method}' method...")
    condition_cliffs = detect_cliffs(args.results_dir, method=args.method, threshold=args.threshold)

    if not condition_cliffs:
        print("No valid data found to analyze.")
        return

    print("\n" + "="*50)
    print_summary_stats(condition_cliffs)

    print("\n" + "="*50)
    run_pairwise_comparisons(condition_cliffs, base_condition=args.base_condition)
    print("="*50)

if __name__ == "__main__":
    main()
