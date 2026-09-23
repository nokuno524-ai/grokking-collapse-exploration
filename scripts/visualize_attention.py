"""
Script to extract and visualize attention patterns across training.
Generates per-layer/head heatmaps and token-position aggregates,
and produces a small HTML gallery for browsing.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.attention_extraction import extract_attention, compute_entropy
from src.model import ModularArithmeticTransformer
from src.data import generate_modular_arithmetic
from src.analysis.grok_detector.detectors import plateau_detector
import matplotlib.animation as animation

def generate_attention_heatmaps(
    attention_weights: List[np.ndarray],
    output_dir: Path,
    step: int,
    tokens: List[str]
):
    """
    Generate and save per-layer/head heatmaps.

    Args:
        attention_weights: List of arrays (batch, heads, seq, seq). One per layer.
        output_dir: Directory to save plots.
        step: Training step (for titles).
        tokens: Token labels for axes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # We average over the batch dimension
    for layer_idx, layer_attn in enumerate(attention_weights):
        # layer_attn shape: (batch, heads, seq, seq)
        mean_attn = layer_attn.mean(axis=0) # (heads, seq, seq)
        num_heads = mean_attn.shape[0]

        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
        if num_heads == 1:
            axes = [axes]

        for head_idx, ax in enumerate(axes):
            attn_matrix = mean_attn[head_idx]

            im = ax.imshow(attn_matrix, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'Layer {layer_idx} Head {head_idx}')

            ax.set_xticks(np.arange(len(tokens)))
            ax.set_yticks(np.arange(len(tokens)))
            ax.set_xticklabels(tokens)
            ax.set_yticklabels(tokens)

            # Add colorbar
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'Attention Maps - Step {step}', fontsize=16)
        plt.tight_layout()

        plot_path = output_dir / f"attention_l{layer_idx}_step{step}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

def generate_entropy_plot(
    entropies: Dict[int, List[float]],
    output_dir: Path
):
    """
    Generate a plot of average attention entropy over training steps.
    """
    if not entropies:
        return

    steps = sorted(list(entropies.keys()))
    avg_entropies = [np.mean(entropies[step]) for step in steps]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, avg_entropies, marker='o', linestyle='-')
    plt.title('Average Attention Entropy Across Training')
    plt.xlabel('Training Step')
    plt.ylabel('Entropy')
    plt.grid(True, alpha=0.3)

    plot_path = output_dir / "attention_entropy.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_html_gallery(output_dir: Path, steps: List[int], layers: int, grok_step: int = None):
    """
    Generate an HTML gallery to browse the generated heatmaps.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attention Visualization Gallery</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            .gallery {{ display: flex; flex-direction: column; gap: 20px; }}
            .step-section {{ border: 1px solid #ccc; padding: 15px; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <h1>Attention Visualization Gallery</h1>
        <div class="gallery">
    """

    # Add entropy plot if it exists
    if (output_dir / "attention_entropy.png").exists():
        html_content += """
        <div class="step-section">
            <h2>Attention Entropy</h2>
            <img src="attention_entropy.png" alt="Attention Entropy Plot">
        </div>
        """

    if (output_dir / "attention_animation.gif").exists():
        html_content += """
        <div class="step-section">
            <h2>Attention Animation</h2>
            <img src="attention_animation.gif" alt="Attention Animation over Training">
        </div>
        """

    for step in sorted(steps):
        highlight_style = 'style="border: 2px solid red; background-color: #ffe6e6;"' if step == grok_step else ""
        html_content += f"""
        <div class="step-section" {highlight_style}>
            <h2>Step {step} { "(GROKKING CLIFF)" if step == grok_step else "" }</h2>
        """
        for layer in range(layers):
            img_name = f"attention_l{layer}_step{step}.png"
            if (output_dir / img_name).exists():
                html_content += f"""
                <h3>Layer {layer}</h3>
                <img src="{img_name}" alt="Attention Heatmap L{layer} Step {step}">
                """
        html_content += "</div>"

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(output_dir / "gallery.html", "w") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Visualize attention patterns.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to the training run directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save visualizations.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cuda/cpu).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate evaluation data (modular arithmetic: a + b = c mod p)
    # Using small prime for fast evaluation
    prime = 59
    X_train, y_train, X_test, y_test = generate_modular_arithmetic(prime)

    # Take a small batch for visualization
    # We want to see how it attends to operands
    batch_size = 32
    X_batch = X_test[:batch_size].to(args.device)

    entropies_over_time = {}
    processed_steps = []
    num_layers = 0

    # Check for grokking step
    grok_step = None
    results_file = args.run_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                results_data = json.load(f)

            history = results_data.get("history", [])
            if history:
                steps_arr = np.array([h["step"] for h in history])
                accs_arr = np.array([h.get("test_acc", h.get("accuracy", 0)) for h in history])
                grok_step = plateau_detector(steps_arr, accs_arr)
                if grok_step is not None:
                    print(f"Detected grokking step at {grok_step}")
        except Exception as e:
            print(f"Failed to load or parse results.json for grokking step: {e}")

    # Find all checkpoints
    checkpoints = sorted([p for p in args.run_dir.glob("checkpoint_*.pt")],
                         key=lambda p: int(p.stem.split("_")[1]))

    if not checkpoints:
        print(f"No checkpoints found in {args.run_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints.")

    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split("_")[1])
        processed_steps.append(step)

        print(f"Processing step {step}...")

        # Load model
        model = ModularArithmeticTransformer(prime=prime)
        state_dict = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.eval()

        with extract_attention(model) as attention_weights:
            with torch.no_grad():
                _ = model(X_batch)

        num_layers = len(attention_weights)

        # Generate heatmaps
        # The sequence length is determined dynamically by the input
        seq_len = X_batch.shape[1]
        if seq_len == 2:
            tokens = ['a', 'b']
        else:
            tokens = [f'tok_{i}' for i in range(seq_len)]

        generate_attention_heatmaps(attention_weights, args.output_dir, step, tokens)

        # Compute and track entropy
        step_entropies = []
        for layer_attn in attention_weights:
            # layer_attn shape: (batch, heads, seq, seq)
            entropy = compute_entropy(layer_attn) # (batch, heads, seq)
            step_entropies.append(entropy.mean().item())

        entropies_over_time[step] = step_entropies

    # Generate animation using the saved heatmaps (Layer 0)
    print("Generating attention animation GIF...")
    try:
        import imageio.v3 as iio
        import imageio
        images = []
        for step in sorted(processed_steps):
            img_path = args.output_dir / f"attention_l0_step{step}.png"
            if img_path.exists():
                images.append(iio.imread(img_path))

        if images:
            imageio.mimsave(args.output_dir / "attention_animation.gif", images, duration=0.5) # 500ms per frame
            print("Successfully saved attention_animation.gif")
    except Exception as e:
        print(f"Failed to generate animation (imageio might be missing): {e}")

    # Generate summary entropy plot
    generate_entropy_plot(entropies_over_time, args.output_dir)

    # Generate HTML gallery
    generate_html_gallery(args.output_dir, processed_steps, num_layers, grok_step)

    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
