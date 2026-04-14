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


def plot_training_trajectory(results_dir: Path, output_path: Optional[Path] = None):
    """Plot training trajectories for all conditions."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    metrics = [
        ("train_loss", "Train Loss"),
        ("test_loss", "Test Loss"),
        ("test_acc", "Test Accuracy"),
        ("weight_norm", "Weight Norm"),
        ("embedding_rank", "Embedding Rank"),
        ("fourier_concentration", "Fourier Concentration"),
        ("mode_collapse", "Mode Collapse"),
        ("kl_div", "KL Divergence"),
        ("memorization", "Memorization Score"),
    ]
    
    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }
    
    for ax, (metric, title) in zip(axes.flat, metrics):
        for condition_dir in sorted(results_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            try:
                with open(condition_dir / "results.json") as f:
                    data = json.load(f)
                history = data.get("history", [])
                if not history:
                    continue
                
                steps = [e["step"] for e in history]
                values = [e.get(metric, 0) for e in history]
                color = colors.get(condition_dir.name, "gray")
                ax.plot(steps, values, label=condition_dir.name, color=color, alpha=0.8)
            except Exception:
                pass
        
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
    
    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        try:
            with open(condition_dir / "results.json") as f:
                data = json.load(f)
            conditions.append(condition_dir.name.replace("_", "\n"))
            grokking_steps.append(data.get("grokking_step") or 0)
            test_accs.append(data.get("final_test_acc", 0))
            fourier_concs.append(data.get("final_fourier_concentration", 0))
        except Exception:
            pass
    
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
