from typing import Dict, List, Any, Optional
from pathlib import Path
import json

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates run configurations and final stats to generate tabular data.
    """
    agg = {}
    for r in runs:
        cond = r.get("condition", "unknown")
        if cond not in agg:
            agg[cond] = []
        agg[cond].append(r)

    summary = {}
    for cond, group in agg.items():
        test_accs = [r.get("final_test_acc") for r in group if r.get("final_test_acc") is not None]
        fouriers = [r.get("final_fourier_concentration") for r in group if r.get("final_fourier_concentration") is not None]
        grok_steps = [r.get("grokking_step") for r in group if r.get("grokking_step") is not None]

        summary[cond] = {
            "n": len(group),
            "test_acc_mean": sum(test_accs) / len(test_accs) if test_accs else float('nan'),
            "fourier_mean": sum(fouriers) / len(fouriers) if fouriers else float('nan'),
            "grok_step_median": sorted(grok_steps)[len(grok_steps)//2] if grok_steps else None,
            "grok_rate": sum(1 for r in group if r.get("grokked", False)) / len(group) if group else 0.0
        }
    return summary

def generate_markdown_table(summary: Dict[str, Any]) -> str:
    """
    Generate a markdown table from summary statistics.
    """
    lines = []
    lines.append("| Condition | N | Test Acc | Fourier | Median Grok Step | Grok Rate |")
    lines.append("|---|---|---|---|---|---|")

    for cond in sorted(summary.keys()):
        s = summary[cond]
        acc = f"{s['test_acc_mean']:.3f}" if not np.isnan(s['test_acc_mean']) else "NaN"
        fourier = f"{s['fourier_mean']:.3f}" if not np.isnan(s['fourier_mean']) else "NaN"
        step = str(s['grok_step_median']) if s['grok_step_median'] is not None else "—"
        rate = f"{s['grok_rate']:.2f}"

        lines.append(f"| {cond} | {s['n']} | {acc} | {fourier} | {step} | {rate} |")

    return "\n".join(lines)

def plot_training_trajectory(runs: List[Dict[str, Any]], output_path: Path):
    """
    Plot training trajectories for collected runs.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("train_loss", "Train Loss"),
        ("test_loss", "Test Loss"),
        ("test_acc", "Test Accuracy"),
        ("weight_norm", "Weight Norm"),
        ("embedding_rank", "Embedding Rank"),
        ("fourier_concentration", "Fourier Concentration"),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics):
        for run in runs:
            history = run.get("history", [])
            if not history:
                continue

            steps = [e["step"] for e in history if metric in e and not np.isnan(e[metric])]
            values = [e[metric] for e in history if metric in e and not np.isnan(e[metric])]

            cond = run.get("condition", "unknown")
            ax.plot(steps, values, label=cond, alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel("Step")

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_grokking_comparison(runs: List[Dict[str, Any]], output_path: Path):
    """Generate a bar chart comparing grokking outcomes."""
    if not HAS_MATPLOTLIB:
        return

    conditions = []
    grokking_steps = []
    test_accs = []
    fourier_concs = []

    # Sort or extract from runs
    for run in runs:
        conditions.append(run.get("condition", "unknown").replace("_", "\n"))
        grokking_steps.append(run.get("grokking_step") or 0)
        test_accs.append(run.get("final_test_acc", 0))
        fourier_concs.append(run.get("final_fourier_concentration", 0))

    if not conditions:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#8e44ad"]
    c_list = [colors[i % len(colors)] for i in range(len(conditions))]

    axes[0].bar(conditions, test_accs, color=c_list)
    axes[0].set_title("Final Test Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Grokking threshold')
    axes[0].legend()

    axes[1].bar(conditions, fourier_concs, color=c_list)
    axes[1].set_title("Fourier Concentration")

    non_zero = [s if s > 0 else 0 for s in grokking_steps]
    axes[2].bar(conditions, non_zero, color=c_list)
    axes[2].set_title("Grokking Step")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")
