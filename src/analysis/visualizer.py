import json
import pathlib
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .parser import load_experiment, scan_results_dir, detect_grokking_point

def plot_loss_curves(df: pd.DataFrame, output_path: str):
    """Train/val loss with grokking point annotated."""
    plt.figure(figsize=(10, 6))

    if "step" not in df.columns or "train_loss" not in df.columns or "test_loss" not in df.columns:
        print("Warning: Missing required columns for plot_loss_curves")
        return

    plt.plot(df["step"], df["train_loss"], label="Train Loss")
    plt.plot(df["step"], df["test_loss"], label="Val Loss")

    # Try to find grokking point
    try:
        # Assuming we're using test_acc or val_acc
        acc_col = "test_acc" if "test_acc" in df.columns else "val_acc"
        if acc_col in df.columns:
            grok_step = detect_grokking_point(df, acc_col=acc_col, threshold=0.95)
            if grok_step != -1:
                plt.axvline(x=grok_step, color='r', linestyle='--', label=f'Grokking (step {grok_step})')
    except Exception:
        pass

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_curves(df: pd.DataFrame, output_path: str):
    """With phase labels (memorization then grokking)."""
    plt.figure(figsize=(10, 6))

    if "step" not in df.columns:
        return

    train_acc_col = "train_acc" if "train_acc" in df.columns else "acc"
    val_acc_col = "test_acc" if "test_acc" in df.columns else "val_acc"

    if train_acc_col in df.columns:
        plt.plot(df["step"], df[train_acc_col], label="Train Accuracy", color="blue")
    if val_acc_col in df.columns:
        plt.plot(df["step"], df[val_acc_col], label="Val Accuracy", color="orange")

    try:
        if val_acc_col in df.columns:
            grok_step = detect_grokking_point(df, acc_col=val_acc_col, threshold=0.95)
            if grok_step != -1:
                plt.axvline(x=grok_step, color='r', linestyle='--', label=f'Grokking (step {grok_step})')

                # Add phase labels
                plt.text(grok_step / 2, 0.5, "Memorization Phase", horizontalalignment='center')
                max_step = df["step"].max()
                plt.text(grok_step + (max_step - grok_step) / 2, 0.5, "Generalization Phase", horizontalalignment='center')
    except Exception:
        pass

    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_collapse_comparison(runs: Dict[str, pd.DataFrame], output_dir: str):
    """Compare collapse levels side-by-side."""
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Compare Accuracy
    plt.figure(figsize=(12, 7))
    for name, df in runs.items():
        if "step" in df.columns:
            val_col = "test_acc" if "test_acc" in df.columns else "val_acc"
            if val_col in df.columns:
                plt.plot(df["step"], df[val_col], label=name)

    plt.xlabel("Step")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Across Collapse Levels")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path / "collapse_comparison_acc.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Compare Weight Norm
    plt.figure(figsize=(12, 7))
    for name, df in runs.items():
        if "step" in df.columns and "weight_norm" in df.columns:
            plt.plot(df["step"], df["weight_norm"], label=name)

    plt.xlabel("Step")
    plt.ylabel("Weight Norm")
    plt.title("Weight Norm Across Collapse Levels")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path / "collapse_comparison_norm.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_weight_norm_trajectory(df: pd.DataFrame, output_path: str):
    """Weight norm evolution."""
    if "step" not in df.columns or "weight_norm" not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["weight_norm"], label="Weight Norm", color="green")

    plt.xlabel("Step")
    plt.ylabel("Weight Norm")
    plt.title("Weight Norm Trajectory")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_heatmap(attention_weights: np.ndarray, output_path: str):
    """Single attention head heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.title("Attention Heatmap")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_evolution(snapshots: List[np.ndarray], output_path: str):
    """Multi-panel attention at different steps."""
    n_snapshots = len(snapshots)
    if n_snapshots == 0:
        return

    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))
    if n_snapshots == 1:
        axes = [axes]

    for i, (ax, attn) in enumerate(zip(axes, snapshots)):
        im = ax.imshow(attn, cmap="viridis", aspect="auto")
        ax.set_title(f"Snapshot {i+1}")
        if i == 0:
            ax.set_ylabel("Query")
        ax.set_xlabel("Key")

    fig.colorbar(im, ax=axes, label="Attention Weight", fraction=0.046, pad=0.04)
    plt.suptitle("Attention Evolution")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_experiment_report(results_dir: str, output_dir: str):
    """Generate all plots plus index.html."""
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    catalog = scan_results_dir(results_dir)
    runs = {}
    html_content = ["<html><head><title>Experiment Report</title><style>body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; } img { max-width: 100%; height: auto; }</style></head><body>"]
    html_content.append("<h1>Grokking Collapse Experiment Report</h1>")

    html_content.append("<h2>Individual Conditions</h2>")

    for entry in catalog:
        name = entry["condition_name"]
        try:
            data = load_experiment(name, results_dir)
            if "history" in data and len(data["history"]) > 0:
                df = pd.DataFrame(data["history"])
                runs[name] = df

                # Generate individual plots
                cond_dir = out_path / name
                cond_dir.mkdir(exist_ok=True)

                plot_loss_curves(df, str(cond_dir / "loss.png"))
                plot_accuracy_curves(df, str(cond_dir / "accuracy.png"))
                plot_weight_norm_trajectory(df, str(cond_dir / "weight_norm.png"))

                html_content.append(f"<h3>Condition: {name}</h3>")
                html_content.append(f"<p>Grokked: {data.get('grokked', False)}, Final Acc: {data.get('final_test_acc', 0.0):.4f}</p>")
                html_content.append("<div style='display: flex; flex-wrap: wrap;'>")
                html_content.append(f"<div style='flex: 33%; padding: 5px;'><img src='{name}/loss.png'></div>")
                html_content.append(f"<div style='flex: 33%; padding: 5px;'><img src='{name}/accuracy.png'></div>")
                html_content.append(f"<div style='flex: 33%; padding: 5px;'><img src='{name}/weight_norm.png'></div>")
                html_content.append("</div>")

        except Exception as e:
            print(f"Failed to process {name}: {e}")

    if runs:
        html_content.append("<h2>Comparisons</h2>")
        plot_collapse_comparison(runs, str(out_path))
        html_content.append("<div style='display: flex; flex-wrap: wrap;'>")
        html_content.append(f"<div style='flex: 50%; padding: 5px;'><img src='collapse_comparison_acc.png'></div>")
        html_content.append(f"<div style='flex: 50%; padding: 5px;'><img src='collapse_comparison_norm.png'></div>")
        html_content.append("</div>")

    html_content.append("</body></html>")

    with open(out_path / "index.html", "w") as f:
        f.write("\n".join(html_content))

    print(f"Report generated at {out_path / 'index.html'}")
