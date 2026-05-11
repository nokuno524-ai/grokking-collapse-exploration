import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def detect_grokking_point(steps: List[int], accuracies: List[float], threshold: float = 0.9) -> Optional[int]:
    """
    Find the first step where accuracy exceeds the threshold and stays above for at least 50 steps.

    Args:
        steps: List of training steps.
        accuracies: List of corresponding accuracies.
        threshold: The accuracy threshold to indicate grokking.

    Returns:
        The step at which grokking is detected, or None if not found.
    """
    for i in range(len(accuracies)):
        if accuracies[i] >= threshold:
            # Check if it stays above for 50 steps
            # "stays above for 50+ steps" means for any j where steps[j] <= steps[i] + 50, accuracy >= threshold
            # Actually, standard interpretation: from step steps[i] to steps[i] + 50, all evaluated points are >= threshold,
            # or if the final step is < steps[i] + 50, it stays above until the end.
            stays_above = True
            # We must reach at least step[i] + 50 to confirm grokking
            reached_50_steps = False
            for j in range(i, len(accuracies)):
                if accuracies[j] < threshold:
                    stays_above = False
                    break
                if steps[j] >= steps[i] + 50:
                    reached_50_steps = True
                    break

            if stays_above and reached_50_steps:
                return steps[i]
    return None


def _load_results(results_dir: str) -> Dict[str, dict]:
    """
    Load results.json from all condition subdirectories.

    Args:
        results_dir: The root directory containing condition subdirectories.

    Returns:
        A dictionary mapping condition name to its loaded JSON data.
    """
    results = {}
    if not os.path.exists(results_dir):
        return results

    for condition in os.listdir(results_dir):
        cond_dir = os.path.join(results_dir, condition)
        if os.path.isdir(cond_dir):
            results_file = os.path.join(cond_dir, "results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    try:
                        results[condition] = json.load(f)
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse JSON in {results_file}")
    return results


def plot_training_curves(results_dir: str, output_path: str):
    """
    Plot loss and accuracy curves for each condition on the same axes.

    Args:
        results_dir: Directory containing results.
        output_path: Path to save the generated plot.
    """
    results = _load_results(results_dir)
    if not results:
        logging.warning("No results found to plot training curves.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    metrics = [("train_loss", "Train Loss"),
               ("test_loss", "Test Loss"),
               ("train_acc", "Train Accuracy"),
               ("test_acc", "Test Accuracy")]

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, (metric_key, metric_name) in enumerate(metrics):
        ax = axs[i]

        for (condition, data), color in zip(results.items(), colors):
            history = data.get("history", [])
            if not history:
                continue

            steps = [entry["step"] for entry in history if metric_key in entry]
            values = [entry[metric_key] for entry in history if metric_key in entry]

            if steps and values:
                ax.plot(steps, values, label=condition, color=color)

            # Add vertical line at grokking point if detected in the data
            if i == 3:  # Only add to Test Accuracy plot for clarity
                # Recalculate grokking point here in case it's not saved in JSON
                test_accs_all = [entry.get("test_acc") for entry in history if "test_acc" in entry and "step" in entry]
                steps_all = [entry.get("step") for entry in history if "test_acc" in entry and "step" in entry]
                grok_step = detect_grokking_point(steps_all, test_accs_all)

                if grok_step is not None:
                    ax.axvline(x=grok_step, color=color, linestyle='--', alpha=0.5)

        ax.set_title(metric_name)
        ax.set_xlabel("Steps")
        ax.set_ylabel(metric_name)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_weight_norm_evolution(results_dir: str, output_path: str):
    """
    Plot weight norm over training steps per condition.

    Args:
        results_dir: Directory containing results.
        output_path: Path to save the generated plot.
    """
    results = _load_results(results_dir)
    if not results:
        logging.warning("No results found to plot weight norm evolution.")
        return

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (condition, data), color in zip(results.items(), colors):
        history = data.get("history", [])
        if not history:
            continue

        steps = [entry["step"] for entry in history if "weight_norm" in entry]
        values = [entry["weight_norm"] for entry in history if "weight_norm" in entry]

        if steps and values:
            plt.plot(steps, values, label=condition, color=color)

    plt.title("Weight Norm Evolution")
    plt.xlabel("Steps")
    plt.ylabel("Weight Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_collapse_comparison(results_dir: str, output_path: str):
    """
    Create a bar chart showing final test and train accuracy by collapse level, grouped by metric.

    Args:
        results_dir: Directory containing results.
        output_path: Path to save the generated plot.
    """
    results = _load_results(results_dir)
    if not results:
        logging.warning("No results found to plot collapse comparison.")
        return

    conditions = []
    final_test_accs = []
    final_train_accs = []

    # Try to group/sort by collapse level from config if available, otherwise sort by condition name
    def get_sort_key(item):
        cond, data = item
        return data.get("config", {}).get("collapse_level", cond)

    sorted_results = sorted(results.items(), key=get_sort_key)

    for condition, data in sorted_results:
        final_test_acc = data.get("final_test_acc", 0.0)
        final_train_acc = data.get("final_train_acc", 0.0)

        # If not at top level, try to find in history
        if "final_test_acc" not in data and "history" in data and data["history"]:
            final_test_acc = data["history"][-1].get("test_acc", 0.0)
            final_train_acc = data["history"][-1].get("train_acc", 0.0)

        conditions.append(condition)
        final_test_accs.append(final_test_acc)
        final_train_accs.append(final_train_acc)

    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(conditions))
    width = 0.35

    plt.bar(x_pos - width/2, final_train_accs, width, label='Train Accuracy', color='skyblue')
    plt.bar(x_pos + width/2, final_test_accs, width, label='Test Accuracy', color='salmon')

    plt.xticks(x_pos, conditions, rotation=45, ha="right")
    plt.title("Final Accuracy by Condition")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_results(results_dir: str) -> dict:
    """
    Aggregate stats per condition including final accuracy, grokking point,
    and weight norm reduction.

    Args:
        results_dir: Directory containing results.

    Returns:
        A dictionary containing aggregated statistics.
    """
    results = _load_results(results_dir)
    analysis = {}

    for condition, data in results.items():
        history = data.get("history", [])

        # Calculate weight norm reduction
        initial_norm = None
        final_norm = None
        if history and "weight_norm" in history[0]:
            initial_norm = history[0]["weight_norm"]
        if history and "weight_norm" in history[-1]:
            final_norm = history[-1]["weight_norm"]

        reduction = None
        if initial_norm is not None and final_norm is not None:
            reduction = initial_norm - final_norm

        # Recalculate grokking point
        steps = []
        test_accs = []
        for entry in history:
            if "step" in entry and "test_acc" in entry:
                steps.append(entry["step"])
                test_accs.append(entry["test_acc"])

        grok_point = detect_grokking_point(steps, test_accs)

        analysis[condition] = {
            "final_test_acc": data.get("final_test_acc"),
            "grokking_point": grok_point,
            "weight_norm_reduction": reduction
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("results_dir", help="Directory containing experiment results")
    parser.add_argument("--out_dir", default="analysis_output", help="Output directory for plots")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    plot_training_curves(args.results_dir, os.path.join(args.out_dir, "training_curves.png"))
    plot_weight_norm_evolution(args.results_dir, os.path.join(args.out_dir, "weight_norm_evolution.png"))
    plot_collapse_comparison(args.results_dir, os.path.join(args.out_dir, "collapse_comparison.png"))

    analysis_stats = analyze_results(args.results_dir)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(analysis_stats, f, indent=2)

    print(f"Generated visualizations and summary in {args.out_dir}")

if __name__ == "__main__":
    main()
