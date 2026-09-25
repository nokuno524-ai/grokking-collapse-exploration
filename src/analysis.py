"""
Visualization and analysis utilities for grokking-collapse experiments.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]


def _ordered_condition_dirs(results_dir: Path) -> List[Path]:
    """Return condition subdirectories in severity order; unknown names go alphabetically at the end.
    Also handles multi-seed structures by yielding inner condition dirs."""
    by_name = {}
    for p in results_dir.iterdir():
        if not p.is_dir():
            continue
        # if this directory contains a results.json, it's a condition dir
        if (p / "results.json").exists():
            by_name[p.name] = p
        else:
            # check if it's a seed directory containing condition dirs
            for sub_p in p.iterdir():
                if sub_p.is_dir() and (sub_p / "results.json").exists():
                    # For plotting purposes, we'll just take the first seed we find if there are multiple.
                    # A better way would be using the new aggregation logic for plotting, but this keeps backwards compatibility.
                    if sub_p.name not in by_name:
                        by_name[sub_p.name] = sub_p

    ordered = [by_name.pop(name) for name in SEVERITY_ORDER if name in by_name]
    ordered.extend(by_name[name] for name in sorted(by_name))
    return ordered


def plot_training_trajectory(results_dir: Path, output_path: Optional[Path] = None):
    """Plot training trajectories for all conditions."""
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
    
    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }
    
    for ax, (metric, title) in zip(axes.flat, metrics):
        for condition_dir in _ordered_condition_dirs(results_dir):
            try:
                with open(condition_dir / "results.json") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                continue

            history = data.get("history", [])
            if not history:
                continue

            steps = []
            values = []
            for e in history:
                val = e.get(metric, np.nan)
                if val is not None and not np.isnan(val):
                    steps.append(e["step"])
                    values.append(val)

            if not steps:
                continue
                
            # deduplicate
            unique_steps = []
            unique_values = []
            seen = set()
            for s, v in zip(steps, values):
                if s not in seen:
                    seen.add(s)
                    unique_steps.append(s)
                    unique_values.append(v)

            color = colors.get(condition_dir.name, "gray")
            ax.plot(unique_steps, unique_values, label=condition_dir.name, color=color, alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = results_dir / "training_trajectories.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_grokking_comparison(results_dir: Path, output_path: Optional[Path] = None):
    """Generate a bar chart comparing grokking outcomes."""
    if not HAS_MATPLOTLIB:
        return
    
    conditions = []
    grokking_steps = []
    test_accs = []
    fourier_concs = []
    
    for condition_dir in _ordered_condition_dirs(results_dir):
        try:
            with open(condition_dir / "results.json") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        conditions.append(condition_dir.name.replace("_", "\n"))
        grokking_steps.append(data.get("grokking_step") or 0)
        test_accs.append(data.get("final_test_acc", 0))
        fourier_concs.append(data.get("final_fourier_concentration", 0))
    
    if not conditions:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(conditions, test_accs, color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#8e44ad"][:len(conditions)])
    axes[0].set_title("Final Test Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Grokking threshold')
    axes[0].legend()
    
    axes[1].bar(conditions, fourier_concs, color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#8e44ad"][:len(conditions)])
    axes[1].set_title("Fourier Concentration")
    
    non_zero = [s if s > 0 else 0 for s in grokking_steps]
    axes[2].bar(conditions, non_zero, color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#8e44ad"][:len(conditions)])
    axes[2].set_title("Grokking Step")
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = results_dir / "grokking_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    import sys
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    plot_training_trajectory(results_dir)
    plot_grokking_comparison(results_dir)
