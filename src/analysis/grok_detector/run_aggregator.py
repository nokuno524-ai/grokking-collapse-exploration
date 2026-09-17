import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

def group_runs_by_severity(results_dir: str | Path) -> Dict[float, List[Dict[str, Any]]]:
    """
    Scans a directory for results.json files and groups the results by collapse severity.

    Args:
        results_dir: Path to the directory containing run output directories.

    Returns:
        A dictionary mapping collapse_severity to a list of run results (dicts).
    """
    results_dir = Path(results_dir)
    grouped_runs = {}

    # We look for results.json in immediate subdirectories, or nested directories (multi-seed structure)
    for res_path in results_dir.rglob("results.json"):
        try:
            with open(res_path, "r") as f:
                res = json.load(f)
        except json.JSONDecodeError:
            continue

        cfg = res.get("config", {})
        if not cfg:
            continue

        severity = cfg.get("collapse_severity")
        if severity is None:
            severity = 0.0 # pure condition

        if severity not in grouped_runs:
            grouped_runs[severity] = []

        grouped_runs[severity].append(res)

    return grouped_runs

def aggregate_and_plot(results_dir: str | Path, output_path: str | Path, max_steps: int = 50000) -> None:
    """
    Groups runs by severity, computes Kaplan-Meier survival curves, plots them,
    and runs statistical tests on the differences.

    Args:
        results_dir: Directory containing results.
        output_path: Path to save the resulting plot.
        max_steps: Maximum number of steps in training.
    """
    import matplotlib.pyplot as plt
    from .stats import compute_kaplan_meier, permutation_test, spearman_correlation_with_ci

    grouped = group_runs_by_severity(results_dir)

    plt.figure(figsize=(10, 6))

    all_severities = sorted(grouped.keys())
    pure_runs = grouped.get(0.0, [])

    # Extract grokking steps for pure runs (using max_steps for right-censored runs)
    pure_grok_steps = np.array([r.get("grokking_step", max_steps) if r.get("grokking_step") is not None else max_steps for r in pure_runs])

    # Collect data for spearman correlation
    all_weight_norms = []
    all_severity_vals = []

    print("\n--- Statistical Analysis ---")

    for severity in all_severities:
        runs = grouped[severity]
        times, probs = compute_kaplan_meier(runs, max_steps)

        # Calculate grokking probability
        grok_prob = np.mean([1 if r.get("grokked") else 0 for r in runs])

        plt.step(times, probs, where='post', label=f'Severity {severity:.2f} (Grok prob: {grok_prob:.2f}, n={len(runs)})')

        # Permutation test against pure
        if severity != 0.0 and len(pure_runs) > 0 and len(runs) > 0:
            severity_grok_steps = np.array([r.get("grokking_step", max_steps) if r.get("grokking_step") is not None else max_steps for r in runs])
            p_value = permutation_test(pure_grok_steps, severity_grok_steps)
            print(f"Permutation test (Pure vs Severity {severity:.2f}): p-value = {p_value:.4f}")

        # Collect weight norms
        for r in runs:
            wn = r.get("final_weight_norm")
            if wn is not None:
                all_weight_norms.append(wn)
                all_severity_vals.append(severity)

    plt.title('Kaplan-Meier Survival Curves for Grokking Step by Severity')
    plt.xlabel('Training Steps')
    plt.ylabel('Probability of NOT Grokking')
    plt.ylim(0, 1.05)
    plt.xlim(0, max_steps)
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSaved Kaplan-Meier plot to {output_path}")

    if len(all_weight_norms) >= 2:
        corr, lower, upper = spearman_correlation_with_ci(all_severity_vals, all_weight_norms)
        print(f"\nSpearman Correlation (Severity vs Weight Norm): {corr:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate runs and plot Kaplan-Meier survival curves.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing run results")
    parser.add_argument("--output", type=str, default="analysis/survival_curves.png", help="Path to save the plot")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum number of training steps")
    args = parser.parse_args()

    aggregate_and_plot(args.results_dir, args.output, args.max_steps)
