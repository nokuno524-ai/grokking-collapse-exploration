import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import torch

from src.attention_extraction import (
    run_attention_extraction,
    compute_entropy,
    compute_diagonal_dominance
)

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    title: str,
    save_path: str | Path,
    input_tokens: Tuple[int, int] | None = None
) -> None:
    """
    Plot and save per-head attention heatmaps for a single sequence.

    Args:
        attn_weights: Shape (num_heads, seq_len, seq_len)
        title: Title of the plot
        save_path: Output file path
        input_tokens: Tuple of (token_a, token_b) for axis labels
    """
    num_heads = attn_weights.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(3 * num_heads, 3))

    if num_heads == 1:
        axes = [axes]

    labels = [f"tok_{t}" for t in input_tokens] if input_tokens else ["a", "b"]

    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(attn_weights[i], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Head {i}")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_step_animation(
    run_dir: str | Path,
    output_path: str | Path,
    sample_idx: int = 0,
    fps: int = 5,
    device: str = "cpu"
) -> None:
    """
    Generate an animation of attention patterns over training steps.

    Args:
        run_dir: Path to the run directory
        output_path: Path to save the animation (.gif or .mp4)
        sample_idx: Index of the batch sample to visualize
        fps: Frames per second for the animation
        device: Device to use for extraction
    """
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))

    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_*.pt in {run_dir}")

    steps = [int(re.findall(r"\d+", p.name)[-1]) for p in ckpts]

    print(f"Generating animation across {len(steps)} steps...")

    # Pre-extract all steps
    history = []
    batch_inputs = None
    prime = None

    for step in steps:
        res = run_attention_extraction(run_dir, step=step, batch_size=16, device=device)

        # Take the first multihead attention module we find
        attn_key = list(res["attention_weights"].keys())[0]
        # Get shape: (batch, heads, seq, seq) -> slice sample -> (heads, seq, seq)
        attn = res["attention_weights"][attn_key][sample_idx]

        history.append(attn)
        if batch_inputs is None:
            batch_inputs = res["batch_inputs"][sample_idx]
            prime = res["prime"]

    # Setup plot
    num_heads = history[0].shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(3 * num_heads, 3.5))
    if num_heads == 1:
        axes = [axes]

    labels = [f"{t}" for t in batch_inputs]

    ims = []
    for i in range(num_heads):
        im = axes[i].imshow(history[0][i], cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(f"Head {i}")
        axes[i].set_xticks(range(2))
        axes[i].set_yticks(range(2))
        axes[i].set_xticklabels(labels)
        axes[i].set_yticklabels(labels)
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        ims.append(im)

    title_text = fig.suptitle(f"Step: {steps[0]}")
    plt.tight_layout()

    def update(frame_idx):
        attn = history[frame_idx]
        for i in range(num_heads):
            ims[i].set_array(attn[i])
        title_text.set_text(f"Step: {steps[frame_idx]}  Input: {batch_inputs[0]}+{batch_inputs[1]} mod {prime}")
        return ims + [title_text]

    ani = animation.FuncAnimation(fig, update, frames=len(steps), blit=False)

    # Save
    out_path = Path(output_path)
    if out_path.suffix == '.gif':
        ani.save(out_path, writer='pillow', fps=fps)
    else:
        ani.save(out_path, fps=fps)
    plt.close()

    print(f"Saved animation to {out_path}")

def extract_metrics_over_time(
    run_dir: str | Path,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Extract attention metrics across all available checkpoints in a run.
    """
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))

    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_*.pt in {run_dir}")

    steps = [int(re.findall(r"\d+", p.name)[-1]) for p in ckpts]

    records = []

    for step in steps:
        res = run_attention_extraction(run_dir, step=step, batch_size=128, device=device)
        attn_key = list(res["attention_weights"].keys())[0]
        # (batch, heads, seq, seq)
        attn = res["attention_weights"][attn_key]

        entropy = compute_entropy(attn)  # (batch, heads, seq)
        diag_dom = compute_diagonal_dominance(attn)  # (batch, heads)

        # Mean across batch
        mean_entropy = np.mean(entropy, axis=0)  # (heads, seq)
        mean_diag = np.mean(diag_dom, axis=0)  # (heads,)

        num_heads = mean_diag.shape[0]
        for h in range(num_heads):
            records.append({
                "step": step,
                "head": h,
                "mean_entropy": np.mean(mean_entropy[h]),
                "diagonal_dominance": mean_diag[h],
                "off_diagonal_mass": 1.0 - mean_diag[h],
                "seed": res["seed"]
            })

    return pd.DataFrame(records)

def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str | Path,
    metric_col: str = "mean_entropy",
    title: str = "Attention Metric over Time"
) -> None:
    """
    Plot attention metric over time per head, with variability bands if multiple seeds are present.
    Expects df with columns: ['step', 'head', metric_col, 'seed']
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_heads = metrics_df["head"].nunique()

    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4), sharey=True)
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        ax = axes[h]
        head_data = metrics_df[metrics_df["head"] == h]

        # Group by step across seeds
        grouped = head_data.groupby("step")[metric_col].agg(['mean', 'std']).reset_index()

        ax.plot(grouped["step"], grouped["mean"], label="Mean")
        if 'std' in grouped.columns and not grouped['std'].isna().all():
            ax.fill_between(
                grouped["step"],
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                alpha=0.2
            )

        ax.set_title(f"Head {h}")
        ax.set_xlabel("Step")
        if h == 0:
            ax.set_ylabel(metric_col.replace("_", " ").title())

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric_col}_over_time.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention extraction and visualization")
    parser.add_argument("--run-dirs", type=str, nargs='+', required=True, help="Path(s) to run directories. If multiple, will group by seed/regime.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--animate", action="store_true", help="Generate animation of first sample for the first run.")
    parser.add_argument("--heatmaps", action="store_true", help="Generate start/mid/end heatmaps for the first run.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for r_dir in args.run_dirs:
        print(f"Extracting metrics over time for {r_dir}...")
        df = extract_metrics_over_time(r_dir, device=args.device)
        # Try to extract a condition/regime name if possible, otherwise use the path name
        df['run_path'] = r_dir

        # Determine regime from config if present, fallback to directory name
        try:
            with open(Path(r_dir) / "results.json") as f:
                res = json.load(f)
                df['regime'] = res.get("condition", Path(r_dir).parent.name)
        except:
            df['regime'] = Path(r_dir).parent.name

        all_metrics.append(df)

    metrics_df = pd.concat(all_metrics, ignore_index=True)

    # Save CSV
    metrics_df.to_csv(output_dir / "attention_metrics.csv", index=False)

    # Plot metrics
    print("Plotting summary metrics...")

    # Group by regime and step
    for metric, title in [("mean_entropy", "Attention Entropy"), ("diagonal_dominance", "Diagonal Dominance")]:
        num_heads = metrics_df["head"].nunique()
        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4), sharey=True)
        if num_heads == 1:
            axes = [axes]

        for h in range(num_heads):
            ax = axes[h]
            head_data = metrics_df[metrics_df["head"] == h]

            for regime in head_data['regime'].unique():
                regime_data = head_data[head_data['regime'] == regime]
                grouped = regime_data.groupby("step")[metric].agg(['mean', 'std']).reset_index()

                ax.plot(grouped["step"], grouped["mean"], label=f"{regime}")
                if 'std' in grouped.columns and not grouped['std'].isna().all():
                    ax.fill_between(
                        grouped["step"],
                        grouped["mean"] - grouped["std"],
                        grouped["mean"] + grouped["std"],
                        alpha=0.2
                    )

            ax.set_title(f"Head {h}")
            ax.set_xlabel("Step")
            if h == 0:
                ax.set_ylabel(title)
                ax.legend()

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_over_time.png", dpi=150)
        plt.close()

    # Optional single-run visuals
    if args.animate or args.heatmaps:
        first_run = args.run_dirs[0]

        if args.heatmaps:
            print("Generating heatmaps...")
            ckpts = sorted(Path(first_run).glob("checkpoint_*.pt"),
                           key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
            if len(ckpts) > 0:
                # Pick start, mid, end
                steps_to_plot = []
                if len(ckpts) >= 1: steps_to_plot.append(int(re.findall(r"\d+", ckpts[0].name)[-1]))
                if len(ckpts) >= 3: steps_to_plot.append(int(re.findall(r"\d+", ckpts[len(ckpts)//2].name)[-1]))
                steps_to_plot.append(int(re.findall(r"\d+", ckpts[-1].name)[-1]))

                # Remove duplicates
                steps_to_plot = sorted(list(set(steps_to_plot)))

                for step in steps_to_plot:
                    res = run_attention_extraction(first_run, step=step, batch_size=1, device=args.device)
                    attn_key = list(res["attention_weights"].keys())[0]
                    attn = res["attention_weights"][attn_key][0]  # First sample
                    inputs = tuple(res["batch_inputs"][0].tolist())

                    plot_attention_heatmap(
                        attn,
                        title=f"Step {step} - Input {inputs[0]}+{inputs[1]}",
                        save_path=output_dir / f"heatmap_step_{step}.png",
                        input_tokens=inputs
                    )

        if args.animate:
            print("Generating animation...")
            generate_step_animation(
                first_run,
                output_dir / "attention_evolution.gif",
                sample_idx=0,
                fps=5,
                device=args.device
            )

    print("Done!")
