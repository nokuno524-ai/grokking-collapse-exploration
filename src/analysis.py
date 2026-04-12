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
    """
    Generate a 2x3 grid of subplots showing how key metrics evolve over the training steps
    for all experimental conditions found in the results directory.

    Plots include Train/Test Loss, Test Accuracy, Weight Norm, Embedding Rank,
    and Fourier Concentration.

    Args:
        results_dir: Directory containing the experiment outputs (folders with results.json).
        output_path: Optional path to save the generated image. Defaults to
                     `results_dir/training_trajectories.png`.
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
    """
    Generate a bar chart comparing the final outcomes of each condition.

    Plots three charts: Final Test Accuracy, Fourier Concentration, and the specific
    step number at which grokking occurred (if at all).

    Args:
        results_dir: Directory containing the experiment outputs.
        output_path: Optional path to save the generated image. Defaults to
                     `results_dir/grokking_comparison.png`.
    """
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


def plot_attention_evolution(results_dir: Path, output_path: Optional[Path] = None, prime: int = 59):
    """
    Extract and plot attention pattern heatmaps over training steps.
    Loads checkpoints across conditions and plots how attention heads evolve,
    showing differences between grokking (pure) and collapsed regimes.

    Args:
        results_dir: Directory containing output condition subdirectories with checkpoints.
        output_path: Optional path to save the generated image.
        prime: Modulus for modular arithmetic to instantiate the model.
    """
    if not HAS_MATPLOTLIB:
        return

    import torch
    import sys
    from pathlib import Path

    # Import the model to load checkpoints
    # Add src to sys.path so we can import model
    src_dir = str(results_dir.parent / "src") if (results_dir.parent / "src").exists() else str(Path("src").absolute())
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    try:
        from model import ModularArithmeticTransformer
    except ImportError:
        print("plot_attention_evolution: Could not import ModularArithmeticTransformer")
        return

    conditions = ["pure", "severe_collapse"] # Just comparing these two for visibility

    # Check if directories exist
    valid_conditions = []
    for cond in conditions:
        if (results_dir / cond).exists():
            valid_conditions.append(cond)

    if not valid_conditions:
        print("plot_attention_evolution: No condition directories found.")
        return

    # Select representative steps (e.g. initial, middle, final)
    target_steps = [5000, 25000, 50000]

    fig, axes = plt.subplots(len(valid_conditions), len(target_steps), figsize=(5 * len(target_steps), 4 * len(valid_conditions)))
    if len(valid_conditions) == 1:
        axes = np.expand_dims(axes, 0)
    if len(target_steps) == 1:
        axes = np.expand_dims(axes, 1)

    device = torch.device("cpu")

    # Example input to evaluate attention on
    test_input = torch.tensor([[0, 1], [prime-1, 1], [prime//2, prime//2]], dtype=torch.long, device=device)

    for i, cond in enumerate(valid_conditions):
        cond_dir = results_dir / cond
        for j, step in enumerate(target_steps):
            ax = axes[i, j]
            ckpt_path = cond_dir / f"checkpoint_{step}.pt"

            if not ckpt_path.exists():
                # Fallback to closest checkpoint or just leave blank
                ckpts = list(cond_dir.glob("checkpoint_*.pt"))
                if not ckpts:
                    ax.axis('off')
                    continue
                ckpt_path = ckpts[-1]

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = ModularArithmeticTransformer(prime=prime).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            with torch.no_grad():
                # Forward pass requesting attention weights
                _, attn_weights = model(test_input, return_attn=True)

                # attn_weights is shape (batch, tgt_len, src_len) = (3, 2, 2)
                # Average over batch to get a general attention pattern
                avg_attn = attn_weights.mean(dim=0).cpu().numpy()

            im = ax.imshow(avg_attn, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"{cond} - Step {ckpt['step']}")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['a', 'b'])
            ax.set_yticklabels(['a', 'b'])

            if j == len(target_steps) - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if output_path is None:
        output_path = results_dir / "attention_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_weight_norm_trajectory(results_dir: Path, output_path: Optional[Path] = None):
    """
    Plot the weight norm trajectory specifically to compare the 'cleanup' phase across conditions.

    Args:
        results_dir: Directory containing output condition subdirectories.
        output_path: Optional path to save the generated image.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

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
            norms = [e.get("weight_norm", 0) for e in history]
            color = colors.get(condition_dir.name, "gray")

            ax.plot(steps, norms, label=condition_dir.name, color=color, linewidth=2)

            # Mark grokking step if available
            grok_step = data.get("grokking_step")
            if grok_step:
                idx = steps.index(grok_step) if grok_step in steps else -1
                if idx != -1:
                    ax.scatter([grok_step], [norms[idx]], color=color, s=100, zorder=5, marker='*')

        except Exception:
            pass

    ax.set_title("Weight Norm Trajectory Over Training")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Weight L2 Norm")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if output_path is None:
        output_path = results_dir / "weight_norm_trajectory.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_loss_landscape_pca(results_dir: Path, output_path: Optional[Path] = None):
    """
    Load historical checkpoints, flatten weight matrices, perform PCA,
    and visualize the 2D trajectory of the model weights over time.

    Args:
        results_dir: Directory containing output condition subdirectories with checkpoints.
        output_path: Optional path to save the generated image.
    """
    if not HAS_MATPLOTLIB:
        return

    import torch
    import sys
    from pathlib import Path
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("plot_loss_landscape_pca: scikit-learn is not installed. Skipping PCA.")
        return

    # To simplify, we'll just track the weights of the 'pure' and 'severe_collapse' conditions.
    conditions = ["pure", "severe_collapse"]

    all_weights = []
    labels = []
    steps = []

    for cond in conditions:
        cond_dir = results_dir / cond
        if not cond_dir.exists():
            continue

        ckpts = sorted(cond_dir.glob("checkpoint_*.pt"), key=lambda x: int(x.stem.split('_')[1]))
        for ckpt_path in ckpts:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt["model_state"]

            # Flatten and concatenate a subset of weights (e.g. embeddings) to track trajectory
            w_emb = state["token_embed.weight"].flatten().numpy()
            w_out = state["output_head.weight"].flatten().numpy()

            w_flat = np.concatenate([w_emb, w_out])
            all_weights.append(w_flat)
            labels.append(cond)
            steps.append(ckpt["step"])

    if not all_weights:
        print("plot_loss_landscape_pca: No checkpoints found.")
        return

    all_weights = np.stack(all_weights)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_weights)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"pure": "#2ecc71", "severe_collapse": "#8e44ad"}

    for cond in conditions:
        mask = [l == cond for l in labels]
        if not any(mask):
            continue

        pts = pca_result[mask]
        s_vals = [s for s, m in zip(steps, mask) if m]

        # Plot trajectory line
        ax.plot(pts[:, 0], pts[:, 1], '-', color=colors[cond], alpha=0.5)

        # Plot points with color gradient based on step
        scatter = ax.scatter(pts[:, 0], pts[:, 1], c=s_vals, cmap='viridis' if cond == 'pure' else 'plasma',
                   edgecolor=colors[cond], label=cond, s=50)

        # Annotate start and end
        ax.annotate('Start', (pts[0, 0], pts[0, 1]), textcoords="offset points", xytext=(0,10), ha='center')
        ax.annotate('End', (pts[-1, 0], pts[-1, 1]), textcoords="offset points", xytext=(0,10), ha='center')

    ax.set_title("PCA of Model Weights Trajectory")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if output_path is None:
        output_path = results_dir / "loss_landscape_pca.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def annotate_phase_transition(results_dir: Path, metric: str = "test_acc", output_path: Optional[Path] = None):
    """
    Plot a specific metric over time for all conditions and automatically annotate
    the phase transition (grokking) point where the metric crosses a threshold.

    Args:
        results_dir: Directory containing experiment outputs.
        metric: The metric to plot and annotate.
        output_path: Optional path to save the generated image.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

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

            ax.plot(steps, values, label=condition_dir.name, color=color, linewidth=2)

            # Use grokking step from results if available
            grok_step = data.get("grokking_step")
            if grok_step and grok_step in steps:
                idx = steps.index(grok_step)
                ax.annotate(
                    f'{condition_dir.name}\nGrok',
                    xy=(grok_step, values[idx]),
                    xytext=(grok_step, values[idx] - 0.2 if values[idx] > 0.5 else values[idx] + 0.2),
                    arrowprops=dict(facecolor=color, shrink=0.05, alpha=0.7),
                    fontsize=9,
                    ha='center'
                )
                ax.scatter([grok_step], [values[idx]], color=color, s=50, zorder=5)

        except Exception:
            pass

    ax.set_title(f"Phase Transition Annotation: {metric}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(metric)
    if metric == "test_acc":
        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.3, label='Grokking threshold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right' if metric == "test_acc" else 'best')

    if output_path is None:
        output_path = results_dir / f"phase_transition_{metric}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def generate_figure_suite(results_dir: Path):
    """
    Template function to load experiment outputs and generate a comprehensive figure suite.
    Runs all visualization functions.

    Args:
        results_dir: Directory containing experiment outputs.
    """
    print(f"Generating figure suite for {results_dir}...")
    plot_training_trajectory(results_dir)
    plot_grokking_comparison(results_dir)
    plot_attention_evolution(results_dir)
    plot_weight_norm_trajectory(results_dir)
    plot_loss_landscape_pca(results_dir)
    annotate_phase_transition(results_dir)
    print("Figure suite generation complete.")

if __name__ == "__main__":
    import sys
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    generate_figure_suite(results_dir)
