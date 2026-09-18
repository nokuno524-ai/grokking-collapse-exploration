"""
Atlas Analysis module.

This module reads the results from the transplant atlas (JSONL format)
and generates analysis artifacts, such as scatter plots of
importance vs rescue effect (with confidence intervals over seeds).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

def read_atlas_results(file_path: Path) -> List[Dict]:
    """
    Reads atlas results from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of result dictionaries.
    """
    results = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def plot_importance_vs_rescue(
    results_list: List[List[Dict]],
    output_path: Path,
    use_rescue: bool = False
):
    """
    Plots a scatter plot of donor importance vs zero-shot or rescue test accuracy.
    Averages over multiple seeds if multiple result lists are provided.

    Args:
        results_list: A list where each element is a list of results (from one seed).
        output_path: Path to save the plot.
        use_rescue: Whether to plot rescue accuracy instead of zero-shot.
    """
    # Aggregate data by component
    comp_to_importance = {}
    comp_to_accs = {}

    for seed_results in results_list:
        for res in seed_results:
            comp = res["component"]
            imp = res["donor_importance"]
            acc = res["rescue_test_acc"] if use_rescue else res["zero_shot_test_acc"]

            if acc is None:
                continue

            if comp not in comp_to_importance:
                comp_to_importance[comp] = []
            if comp not in comp_to_accs:
                comp_to_accs[comp] = []

            comp_to_importance[comp].append(imp)
            comp_to_accs[comp].append(acc)

    components = list(comp_to_importance.keys())

    if not components:
        print("No valid data to plot.")
        return

    avg_imps = [np.mean(comp_to_importance[c]) for c in components]
    avg_accs = [np.mean(comp_to_accs[c]) for c in components]

    # Simple 95% CI (1.96 * std / sqrt(N))
    acc_errs = [1.96 * np.std(comp_to_accs[c]) / np.sqrt(len(comp_to_accs[c])) if len(comp_to_accs[c]) > 1 else 0 for c in components]

    plt.figure(figsize=(10, 6))
    plt.errorbar(avg_imps, avg_accs, yerr=acc_errs, fmt='o', alpha=0.7, ecolor='gray')

    # Annotate points
    for i, c in enumerate(components):
        plt.annotate(c, (avg_imps[i], avg_accs[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xlabel("Donor Component Importance (Activation Patching Delta)")
    plt.ylabel(f"{'Rescue' if use_rescue else 'Zero-Shot'} Test Accuracy")
    plt.title(f"Component Importance vs {'Rescue' if use_rescue else 'Zero-Shot'} Effect")
    plt.grid(True, linestyle='--', alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Testing atlas analysis...")
