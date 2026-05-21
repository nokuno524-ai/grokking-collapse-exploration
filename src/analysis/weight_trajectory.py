"""
Visualization of weight norm trajectories.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

COLORS = {
    "pure": "#2ecc71",
    "low_collapse": "#3498db",
    "medium_collapse": "#f39c12",
    "high_collapse": "#e74c3c",
    "severe_collapse": "#8e44ad",
}


def load_model_from_checkpoint(ckpt_path: Path):
    """Load model from checkpoint."""
    from src.model import ModularArithmeticTransformer
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})

    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1)
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def get_layer_norms(model) -> Dict[str, float]:
    """Calculate L2 norms for distinct parts/layers of the model."""
    norms = {}

    # Token Embedding
    norms["token_embed"] = model.token_embed.weight.norm().item()

    # Positional Embedding
    norms["pos_embed"] = model.pos_embed.weight.norm().item()

    # Transformer Encoder Layers (Combine all params in the transformer)
    t_norm_sq = 0.0
    for name, param in model.transformer.named_parameters():
        t_norm_sq += param.norm().item() ** 2
    norms["transformer"] = t_norm_sq ** 0.5

    # Output Head
    o_norm_sq = 0.0
    for name, param in model.output_head.named_parameters():
        o_norm_sq += param.norm().item() ** 2
    norms["output_head"] = o_norm_sq ** 0.5

    return norms


def plot_weight_trajectories(results_dir: Path, output_path: Path):
    """
    Plot the evolution of weight norms per layer across training
    for each collapse condition. Highlight grokking transition points if present.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    if not results_dir.exists():
        return

    condition_data = {}

    for condition_dir in results_dir.iterdir():
        if not condition_dir.is_dir():
            continue

        condition = condition_dir.name
        results_file = condition_dir / "results.json"

        grok_step = None
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                if data.get("grokked"):
                    grok_step = data.get("grokking_step")
            except Exception:
                pass

        checkpoints = sorted(condition_dir.glob("checkpoint_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
        if not checkpoints:
            continue

        steps = []
        layer_norms = defaultdict(list)

        for ckpt_path in checkpoints:
            try:
                step = int(ckpt_path.stem.split("_")[1])
                model = load_model_from_checkpoint(ckpt_path)
                norms = get_layer_norms(model)

                steps.append(step)
                for layer, norm in norms.items():
                    layer_norms[layer].append(norm)
            except Exception as e:
                print(f"Error processing {ckpt_path}: {e}")

        if steps:
            condition_data[condition] = {
                "steps": steps,
                "layer_norms": layer_norms,
                "grok_step": grok_step
            }

    if not condition_data:
        print("No valid checkpoint data found for weight trajectory plotting.")
        return

    # Find all layer names
    all_layers = set()
    for data in condition_data.values():
        all_layers.update(data["layer_norms"].keys())

    all_layers = sorted(list(all_layers))

    fig, axes = plt.subplots(len(all_layers), 1, figsize=(10, 4 * len(all_layers)), sharex=True)
    if len(all_layers) == 1:
        axes = [axes]

    from matplotlib.lines import Line2D

    for idx, layer in enumerate(all_layers):
        ax = axes[idx]

        for condition, data in condition_data.items():
            if layer not in data["layer_norms"]:
                continue

            color = COLORS.get(condition, 'gray')
            steps = data["steps"]
            norms = data["layer_norms"][layer]

            ax.plot(steps, norms, label=condition.replace("_", " ").title(),
                    color=color, linewidth=2, alpha=0.8)

            grok_step = data["grok_step"]
            if grok_step is not None:
                # Find the nearest step
                nearest_step_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - grok_step))
                nearest_step = steps[nearest_step_idx]
                norm_at_grok = norms[nearest_step_idx]

                ax.scatter([nearest_step], [norm_at_grok], color=color, s=150,
                           edgecolors='black', zorder=5, marker='*')

        ax.set_title(f"{layer.replace('_', ' ').title()} Weight Norm")
        ax.set_ylabel("L2 Norm")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=12))
            labels.append('Grokking Point')
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1))

    axes[-1].set_xlabel("Training Step")

    plt.suptitle("Weight Norm Evolution Per Layer", y=1.02, fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
