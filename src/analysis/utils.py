"""
Visualization and analysis utilities for grokking-collapse experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)

SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

COLORS: Dict[str, str] = {
    "pure": "#2ecc71",
    "low_collapse": "#3498db",
    "medium_collapse": "#f39c12",
    "high_collapse": "#e74c3c",
    "severe_collapse": "#8e44ad",
}
DEFAULT_COLORS: List[str] = list(COLORS.values())


def _ordered_condition_dirs(results_dir: Path) -> List[Path]:
    """Return condition subdirectories in severity order; unknown names go alphabetically at the end."""
    by_name = {p.name: p for p in results_dir.iterdir() if p.is_dir()}
    ordered = [by_name.pop(name) for name in SEVERITY_ORDER if name in by_name]
    ordered.extend(by_name[name] for name in sorted(by_name))
    return ordered


def _load_results_json(condition_dir: Path) -> Optional[Dict[str, Any]]:
    """Helper to load results.json from a condition directory."""
    results_path = condition_dir / "results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {results_path}: {e}")
        return None


def plot_training_trajectory(results_dir: Path, output_path: Optional[Path] = None) -> None:
    """Plot training trajectories for all conditions."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
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
        for condition_dir in _ordered_condition_dirs(results_dir):
            data = _load_results_json(condition_dir)
            if not data:
                continue
                
            history = data.get("history", [])
            if not history:
                continue

            steps = [e["step"] for e in history]
            values = [e.get(metric, 0) for e in history]
            color = COLORS.get(condition_dir.name, "gray")
            ax.plot(steps, values, label=condition_dir.name, color=color, alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = results_dir / "training_trajectories.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to {output_path}")


def plot_grokking_comparison(results_dir: Path, output_path: Optional[Path] = None) -> None:
    """Generate a bar chart comparing grokking outcomes."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    conditions: List[str] = []
    grokking_steps: List[int] = []
    test_accs: List[float] = []
    fourier_concs: List[float] = []
    
    for condition_dir in _ordered_condition_dirs(results_dir):
        data = _load_results_json(condition_dir)
        if not data:
            continue

        conditions.append(condition_dir.name.replace("_", "\n"))
        grokking_steps.append(data.get("grokking_step") or 0)
        test_accs.append(data.get("final_test_acc", 0.0))
        fourier_concs.append(data.get("final_fourier_concentration", 0.0))
    
    if not conditions:
        logger.warning("No condition data found to plot grokking comparison.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    bar_colors = DEFAULT_COLORS[:len(conditions)]

    axes[0].bar(conditions, test_accs, color=bar_colors)
    axes[0].set_title("Final Test Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Grokking threshold')
    axes[0].legend()
    
    axes[1].bar(conditions, fourier_concs, color=bar_colors)
    axes[1].set_title("Fourier Concentration")
    
    non_zero = [s if s > 0 else 0 for s in grokking_steps]
    axes[2].bar(conditions, non_zero, color=bar_colors)
    axes[2].set_title("Grokking Step")
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = results_dir / "grokking_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to {output_path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    res_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    plot_training_trajectory(res_dir)
    plot_grokking_comparison(res_dir)
