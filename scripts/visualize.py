import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add src to sys.path so we can import model/data
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModularArithmeticTransformer

sns.set_theme(style="whitegrid", context="paper")


def load_history(results_dir: Path, condition: str):
    """Load history from results.json."""
    results_path = results_dir / condition / "results.json"
    if not results_path.exists():
        return None
    with open(results_path, "r") as f:
        data = json.load(f)
    return data.get("history", [])


def plot_attention_evolution(results_dir: Path, output_dir: Path, condition: str = "pure", prime: int = 59):
    """(a) Attention pattern evolution heatmaps across training steps."""
    cond_dir = results_dir / condition
    if not cond_dir.exists():
        print(f"Directory {cond_dir} does not exist. Skipping attention heatmaps.")
        return

    checkpoints = sorted(cond_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return

    # Select up to 4 checkpoints for evolution (first, some middle, last)
    indices = np.linspace(0, len(checkpoints) - 1, min(4, len(checkpoints)), dtype=int)
    selected_ckpts = [checkpoints[i] for i in indices]

    fig, axes = plt.subplots(1, len(selected_ckpts), figsize=(5 * len(selected_ckpts), 4))
    if len(selected_ckpts) == 1:
        axes = [axes]

    for ax, ckpt_path in zip(axes, selected_ckpts):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        step = ckpt["step"]

        # Instantiate model and load weights
        model = ModularArithmeticTransformer(prime=prime)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Compute attention pattern for a subset of queries/keys
        # W_Q @ W_K.T approximation from token embedding
        with torch.no_grad():
            W_E = model.token_embed.weight # (prime, d_model)
            # Getting Q and K from the first head of the transformer layer
            mha = model.transformer.layers[0].self_attn
            # in_proj_weight is (3 * d_model, d_model) -> Q, K, V
            d_model = model.d_model
            W_Q = mha.in_proj_weight[:d_model, :]
            W_K = mha.in_proj_weight[d_model:2*d_model, :]

            # Project embeddings
            Q = W_E @ W_Q.T # (prime, d_model)
            K = W_E @ W_K.T # (prime, d_model)

            # Simple attention score approximation across the vocabulary
            attn_scores = Q @ K.T / (d_model ** 0.5)
            attn_probs = torch.softmax(attn_scores, dim=-1)

        sns.heatmap(attn_probs.numpy()[:30, :30], ax=ax, cmap="viridis", cbar=False)
        ax.set_title(f"Step {step}")
        ax.set_xlabel("Key Token (0-29)")
        ax.set_ylabel("Query Token (0-29)")

    plt.tight_layout()
    out_path = output_dir / f"attention_evolution_{condition}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_loss_landscape(results_dir: Path, output_dir: Path):
    """(b) Loss landscape plots comparing collapse vs grokking trajectories."""
    conditions = [d.name for d in results_dir.iterdir() if d.is_dir()]
    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("husl", len(conditions))
    for cond, color in zip(conditions, colors):
        history = load_history(results_dir, cond)
        if not history:
            continue

        steps = [entry["step"] for entry in history]
        test_loss = [entry["test_loss"] for entry in history]
        train_loss = [entry["train_loss"] for entry in history]

        ax.plot(steps, test_loss, label=f"{cond} (Test)", color=color, linestyle="-")
        ax.plot(steps, train_loss, label=f"{cond} (Train)", color=color, linestyle="--", alpha=0.5)

        # Phase transition marker for grokking
        for entry in history:
            if entry.get("test_acc", 0) > 0.95:
                ax.axvline(x=entry["step"], color=color, linestyle=":", alpha=0.8)
                ax.text(entry["step"], max(test_loss)/2, "Grokking", rotation=90, color=color, alpha=0.8)
                break

    ax.set_yscale("log")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss (Log Scale)")
    ax.set_title("Loss Landscape: Collapse vs Grokking")
    ax.legend(loc="upper right", fontsize="small", bbox_to_anchor=(1.35, 1))

    plt.tight_layout()
    out_path = output_dir / "loss_landscape.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_norms_dashboard(results_dir: Path, output_dir: Path):
    """(c) Weight norm tracking dashboard."""
    conditions = [d.name for d in results_dir.iterdir() if d.is_dir()]
    if not conditions:
        return

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    colors = sns.color_palette("Set2", len(conditions))
    for cond, color in zip(conditions, colors):
        history = load_history(results_dir, cond)
        if not history:
            continue

        steps = [entry["step"] for entry in history]
        weight_norm = [entry["weight_norm"] for entry in history]
        fourier = [entry["fourier_concentration"] for entry in history]

        # Approximate gradient norm tracking as loss difference since grad norm isn't consistently stored in basic state
        loss_diffs = np.abs(np.diff([entry["train_loss"] for entry in history], prepend=0))

        axes[0].plot(steps, weight_norm, label=cond, color=color, linewidth=2)
        axes[1].plot(steps, fourier, label=cond, color=color, linewidth=2)
        axes[2].plot(steps, loss_diffs, label=cond, color=color, linewidth=1, alpha=0.7)

    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Total Weight Norm ||W||")
    axes[0].set_title("Weight Norm vs Training Step")
    axes[0].legend()

    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Fourier Concentration")
    axes[1].set_title("Fourier Concentration vs Training Step")
    axes[1].legend()

    axes[2].set_xlabel("Training Steps")
    axes[2].set_ylabel("Approx. Gradient Magnitude (Loss Diff)")
    axes[2].set_title("Training Loss Gradient Proxy vs Training Step")
    axes[2].legend()

    plt.tight_layout()
    out_path = output_dir / "norms_dashboard.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_capability_emergence(results_dir: Path, output_dir: Path):
    """(d) Capability emergence plots (accuracy vs training step)."""
    conditions = [d.name for d in results_dir.iterdir() if d.is_dir()]
    if not conditions:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("Dark2", len(conditions))
    for cond, color in zip(conditions, colors):
        history = load_history(results_dir, cond)
        if not history:
            continue

        steps = [entry["step"] for entry in history]
        test_acc = [entry["test_acc"] for entry in history]

        ax.plot(steps, test_acc, label=cond, color=color, linewidth=2.5)

    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.6, label="Grokking Threshold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Capability Emergence: Test Accuracy vs Step")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = output_dir / "capability_emergence.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Visualizations for Grokking-Collapse.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing experiment results.")
    parser.add_argument("--output-dir", type=str, default="analysis_output/plots", help="Directory to save plots.")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"Error: Results directory '{results_path}' does not exist.")
        return

    print("Generating Visualizations...")
    plot_loss_landscape(results_path, output_path)
    plot_norms_dashboard(results_path, output_path)
    plot_capability_emergence(results_path, output_path)

    # Generate heatmaps for all conditions found
    for cond_dir in results_path.iterdir():
        if cond_dir.is_dir():
            plot_attention_evolution(results_path, output_path, condition=cond_dir.name)

    print(f"All plots saved to: {output_path}")

if __name__ == "__main__":
    main()
