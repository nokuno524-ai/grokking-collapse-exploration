"""
Unified Dashboard for grokking-collapse experiments.
Combines multiple visualizations into a single publication-ready multi-panel figure.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis import _ordered_condition_dirs

def load_training_trajectories(results_dir: Path):
    """Load training trajectory data for all conditions."""
    data = {}
    for condition_dir in _ordered_condition_dirs(results_dir):
        try:
            with open(condition_dir / "results.json") as f:
                res = json.load(f)
            history = res.get("history", [])
            if history:
                data[condition_dir.name] = {
                    "step": [e["step"] for e in history],
                    "test_acc": [e.get("test_acc", 0) for e in history],
                    "weight_norm": [e.get("weight_norm", 0) for e in history],
                }
        except Exception:
            pass
    return data

def plot_trajectory_panel(ax, data, metric, ylabel, title):
    """Plot a trajectory panel."""
    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

    for cond, cond_data in data.items():
        if cond_data[metric]:
            ax.plot(cond_data["step"], cond_data[metric],
                    label=cond.replace("_", " ").title(),
                    color=colors.get(cond, "gray"), linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

def add_image_panel(ax, img_path: Path, title: str):
    """Add an existing image to a panel."""
    if img_path.exists():
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, pad=10)
    else:
        ax.text(0.5, 0.5, f"Image not found:\n{img_path.name}",
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

def create_dashboard(results_dir: Path, viz_output_dir: Path, save_path: Path):
    """Create the unified dashboard figure."""
    # Setup overall figure and styling
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(20, 15))

    # Create grid layout
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.2)

    # 1. Training Trajectories (Row 1, spanning both cols)
    traj_data = load_training_trajectories(results_dir)

    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :], wspace=0.2)
    ax_acc = fig.add_subplot(gs_row1[0, 0])
    ax_norm = fig.add_subplot(gs_row1[0, 1])

    if traj_data:
        plot_trajectory_panel(ax_acc, traj_data, "test_acc", "Test Accuracy", "A) Grokking Delay with Collapse")
        ax_acc.axhline(0.95, color='r', linestyle='--', alpha=0.5)
        ax_acc.legend(fontsize=10, loc='lower right')

        plot_trajectory_panel(ax_norm, traj_data, "weight_norm", "L2 Norm", "B) Weight Norm Evolution")
    else:
        ax_acc.text(0.5, 0.5, "No trajectory data", ha='center', va='center')
        ax_norm.text(0.5, 0.5, "No trajectory data", ha='center', va='center')

    # 2. Phase Diagram and 3D Weight Norms (Row 2)
    ax_phase = fig.add_subplot(gs[1, 0])
    ax_3d = fig.add_subplot(gs[1, 1])

    add_image_panel(ax_phase, viz_output_dir / "phase_diagram_steps.png", "C) Grokking Phase Diagram")
    add_image_panel(ax_3d, viz_output_dir / "weight_norms_3d_token_embed.png", "D) Token Embedding Norm Surface")

    # 3. Attention Comparison and Loss Landscape (Row 3)
    ax_attn = fig.add_subplot(gs[2, 0])
    ax_loss = fig.add_subplot(gs[2, 1])

    add_image_panel(ax_attn, viz_output_dir / "attention_compare.png", "E) Attention Pattern Breakdown")
    add_image_panel(ax_loss, viz_output_dir / "loss_landscape_evolution.png", "F) Loss Landscape Geometry")

    # Finalize and save
    fig.suptitle("Model Collapse Monotonically Prevents Grokking", fontsize=20, y=0.95, fontweight='bold')

    # Save as high-res PNG and PDF for paper
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--viz-dir", type=str, default="viz_output")
    parser.add_argument("--output", type=str, default="viz_output/dashboard_main_figure.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    viz_dir = Path(args.viz_dir)
    output_path = Path(args.output)

    print("Generating unified dashboard figure...")
    create_dashboard(results_dir, viz_dir, output_path)
    print(f"Dashboard saved to {output_path}")
