"""
Plotting Kaplan-Meier survival curves for ensemble sweeps.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.ensemble.aggregation import aggregate_results

def plot_survival_curves(aggregated: dict, output_path: str):
    """
    Plot KM survival curves for each condition.
    Survival probability = probability of NOT grokking by time t.
    Usually we plot 1 - survival (probability of grokking).
    """
    plt.figure(figsize=(10, 6))

    # Sort conditions by median grok step for legend order
    sorted_conds = sorted(aggregated.items(), key=lambda x: x[1]["median_grok_step"])

    for condition, stats in sorted_conds:
        times = np.array(stats["km_times"])
        probs = np.array(stats["km_probs"])
        variances = np.array(stats["km_variances"])

        # Plot probability of grokking (1 - survival)
        grok_probs = 1.0 - probs

        # Calculate 95% confidence intervals
        # Variance of (1 - S) is the same as variance of S
        se = np.sqrt(variances)
        ci_upper = np.clip(grok_probs + 1.96 * se, 0, 1)
        ci_lower = np.clip(grok_probs - 1.96 * se, 0, 1)

        # Plot with step function
        p = plt.step(times, grok_probs, where="post", label=f"{condition} (n={stats['n_runs']})", linewidth=2)
        color = p[0].get_color()

        # Fill between for confidence interval
        plt.fill_between(times, ci_lower, ci_upper, step="post", alpha=0.2, color=color)

    plt.title("Grokking Probability vs. Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Probability of Grokking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    parser.add_argument("--output-plot", type=str, default="analysis/ensemble_survival.png", help="Output plot path")
    parser.add_argument("--max-steps", type=int, default=50000, help="Max steps for censoring")
    args = parser.parse_args()

    agg = aggregate_results(args.results_dir, args.max_steps)
    plot_survival_curves(agg, args.output_plot)
