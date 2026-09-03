import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import defaultdict

try:
    from .grok_detector.stats import kaplan_meier
except ImportError:
    from grok_detector.stats import kaplan_meier

SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

def load_data(jsonl_path):
    data = defaultdict(list)
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            data[entry['condition']].append(entry)
    return data

def plot_grokking_distributions(data, out_dir):
    plt.figure(figsize=(10, 6))

    plot_data = []
    labels = []

    for cond in SEVERITY_ORDER:
        if cond not in data:
            continue
        # Extract grokking steps, ignore Nones (never grokked)
        steps = [d['grokking_step'] for d in data[cond] if d['grokked']]
        plot_data.append(steps)
        labels.append(cond.replace('_', '\n'))

    # We might have empty lists if a condition never groks
    valid_data = [d for d in plot_data if len(d) > 0]
    valid_labels = [labels[i] for i in range(len(plot_data)) if len(plot_data[i]) > 0]

    if valid_data:
        plt.boxplot(valid_data, tick_labels=valid_labels)
        plt.title('Distribution of Grokking Steps by Condition')
        plt.ylabel('Training Step')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(out_dir / 'grokking_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_survival_curves(data, out_dir, max_steps):
    plt.figure(figsize=(10, 6))

    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

    for cond in SEVERITY_ORDER:
        if cond not in data:
            continue

        times = []
        events = []

        for d in data[cond]:
            if d['grokked']:
                times.append(d['grokking_step'])
                events.append(1)
            else:
                times.append(max_steps)
                events.append(0)

        times = np.array(times)
        events = np.array(events)

        ut, surv = kaplan_meier(times, events)

        # Plot step function for survival
        if len(ut) > 0:
            ut_plot = np.insert(ut, 0, 0)
            surv_plot = np.insert(surv, 0, 1.0)
            plt.step(ut_plot, surv_plot, where='post', label=cond, color=colors.get(cond, 'gray'), linewidth=2)

    plt.title('Kaplan-Meier Survival Curves (Probability of NOT Grokking)')
    plt.xlabel('Training Step')
    plt.ylabel('Probability')
    plt.ylim(0, 1.05)
    plt.xlim(0, max_steps)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(out_dir / 'survival_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_final_accuracies(data, out_dir):
    plt.figure(figsize=(10, 6))

    plot_data = []
    labels = []

    for cond in SEVERITY_ORDER:
        if cond not in data:
            continue
        accs = [d['final_test_acc'] for d in data[cond]]
        plot_data.append(accs)
        labels.append(cond.replace('_', '\n'))

    if plot_data:
        plt.violinplot(plot_data, showmeans=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.title('Final Test Accuracy Distribution by Condition')
        plt.ylabel('Test Accuracy')
        plt.grid(axis='y', alpha=0.3)
        plt.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Grokking Threshold')
        plt.legend()
        plt.savefig(out_dir / 'final_accuracies.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_markdown_report(data, out_dir, max_steps):
    report_path = out_dir / 'grokking_report.md'

    with open(report_path, 'w') as f:
        f.write("# Grokking Multi-Seed Statistical Report\n\n")
        f.write("This report presents the results of the multi-seed robustness analysis for model collapse effects on grokking.\n\n")

        f.write("## Summary Statistics\n\n")
        f.write("| Condition | Seeds | Grok Rate | Mean Grok Step | 95% Final Acc | Censored |\n")
        f.write("|-----------|-------|-----------|----------------|---------------|----------|\n")

        for cond in SEVERITY_ORDER:
            if cond not in data:
                continue

            runs = data[cond]
            n_seeds = len(runs)
            n_grokked = sum(1 for r in runs if r['grokked'])
            grok_rate = n_grokked / n_seeds

            grok_steps = [r['grokking_step'] for r in runs if r['grokked']]
            mean_step = np.mean(grok_steps) if grok_steps else float('nan')

            accs = [r['final_test_acc'] for r in runs]
            mean_acc = np.mean(accs)

            f.write(f"| {cond} | {n_seeds} | {grok_rate:.2%} | {mean_step:.1f} | {mean_acc:.3f} | {n_seeds - n_grokked} |\n")

        f.write("\n## Power Note\n\n")
        f.write("To distinguish adjacent conditions (e.g., pure vs low_collapse) at α=0.05 with 80% power:\n")

        # Simple power calculation based on proportions (if applicable) or Cohen's d for continuous steps
        pure = [r['grokking_step'] for r in data.get('pure', []) if r['grokked']]
        low = [r['grokking_step'] for r in data.get('low_collapse', []) if r['grokked']]

        if len(pure) > 1 and len(low) > 1:
            mean1, std1 = np.mean(pure), np.std(pure, ddof=1)
            mean2, std2 = np.mean(low), np.std(low, ddof=1)
            # Pooled standard deviation
            pooled_std = np.sqrt(((len(pure)-1)*std1**2 + (len(low)-1)*std2**2) / (len(pure) + len(low) - 2))
            if pooled_std > 0:
                d = abs(mean1 - mean2) / pooled_std
                # Approximation for n per group for 80% power at alpha=0.05 is ~ 16 / d^2
                if d > 0:
                    n_needed = int(np.ceil(16 / (d**2)))
                    f.write(f"- Effect size (Cohen's d) between `pure` and `low_collapse`: {d:.2f}\n")
                    f.write(f"- Estimated sample size needed per group: **{n_needed} seeds**.\n")
                else:
                    f.write("- Effect size is ~0; very large sample size needed.\n")
            else:
                 f.write("- Variance is zero; unable to estimate power.\n")
        else:
            f.write("- Insufficient grokking events to compute effect size between `pure` and `low_collapse`.\n")

        f.write("\n## Visualizations\n")
        f.write("- ![Grokking Distributions](grokking_distributions.png)\n")
        f.write("- ![Survival Curves](survival_curves.png)\n")
        f.write("- ![Final Accuracies](final_accuracies.png)\n")

    return report_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="results/multi_seed_stats.jsonl")
    parser.add_argument("--out-dir", type=str, default="analysis/multi_seed")
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} not found.")
        return

    data = load_data(args.input_file)

    if not data:
        print("No data found.")
        return

    print("Generating visualizations...")
    plot_grokking_distributions(data, out_dir)
    plot_survival_curves(data, out_dir, args.max_steps)
    plot_final_accuracies(data, out_dir)

    print("Generating markdown report...")
    report_path = generate_markdown_report(data, out_dir, args.max_steps)

    print(f"Report complete: {report_path}")

if __name__ == "__main__":
    main()
