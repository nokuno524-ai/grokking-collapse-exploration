"""
Weight norm trajectory visualization for grokking-collapse experiments.
Calculates L2 norms of parameters across training steps and models,
producing faceted line charts and a 3D surface plot.
"""

import os
import torch
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis import _ordered_condition_dirs

def get_layer_norms(state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute L2 norms for the key layers in the ModularArithmeticTransformer state dict.
    Returns a dictionary mapping layer names to their L2 norm.
    """
    norms = {}

    # Embedding layer
    if "token_embed.weight" in state_dict:
        norms["token_embed"] = state_dict["token_embed.weight"].float().norm().item()

    # Positional embedding
    if "pos_embed.weight" in state_dict:
        norms["pos_embed"] = state_dict["pos_embed.weight"].float().norm().item()

    # Transformer layers (Q, K, V, Out projections, MLP)
    layer_idx = 0
    while True:
        # Check if layer exists
        q_proj = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
        if q_proj not in state_dict:
            break

        # Group transformer layer weights
        # self attention in_proj (Q, K, V concatenated)
        attn_weight = state_dict[q_proj].float()
        norms[f"layer_{layer_idx}_attn_in"] = attn_weight.norm().item()

        # out projection
        out_proj = state_dict[f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"].float()
        norms[f"layer_{layer_idx}_attn_out"] = out_proj.norm().item()

        # MLP
        linear1 = state_dict[f"transformer.layers.{layer_idx}.linear1.weight"].float()
        norms[f"layer_{layer_idx}_mlp1"] = linear1.norm().item()

        linear2 = state_dict[f"transformer.layers.{layer_idx}.linear2.weight"].float()
        norms[f"layer_{layer_idx}_mlp2"] = linear2.norm().item()

        layer_idx += 1

    # Output head
    if "output_head.weight" in state_dict:
        norms["output_head"] = state_dict["output_head.weight"].float().norm().item()

    return norms

def collect_norm_trajectories(results_dir: Path) -> Dict[str, Dict[str, List[tuple]]]:
    """
    Collects layer weight norms over training steps for all conditions.
    Returns: { condition_name: { layer_name: [(step, norm), ...] } }
    """
    condition_dirs = _ordered_condition_dirs(results_dir)
    all_data = {}

    for condition_dir in condition_dirs:
        condition_name = condition_dir.name
        all_data[condition_name] = {}

        checkpoints = sorted(list(condition_dir.glob("checkpoint_*.pt")),
                             key=lambda p: int(p.stem.split("_")[1]))

        for ckpt_path in checkpoints:
            try:
                step = int(ckpt_path.stem.split("_")[1])
                ckpt = torch.load(ckpt_path, map_location="cpu")
                norms = get_layer_norms(ckpt["model_state"])

                for layer, norm in norms.items():
                    if layer not in all_data[condition_name]:
                        all_data[condition_name][layer] = []
                    all_data[condition_name][layer].append((step, norm))
            except Exception as e:
                print(f"Error loading {ckpt_path}: {e}")

    # Sort lists by step
    for cond_data in all_data.values():
        for layer_data in cond_data.values():
            layer_data.sort(key=lambda x: x[0])

    return all_data

def plot_faceted_norms(all_data: Dict[str, Dict[str, List[tuple]]], save_path: Path):
    """
    Plot layer norms over time, faceted by condition (collapse level).
    """
    conditions = list(all_data.keys())
    if not conditions:
        print("No data available for faceted norm plot.")
        return

    # Find common layers across all conditions
    sample_cond = conditions[0]
    layers = list(all_data[sample_cond].keys())

    n_cols = len(conditions)
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6), sharey=True)
    if n_cols == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))

    for idx, (cond, ax) in enumerate(zip(conditions, axes)):
        for layer_idx, layer in enumerate(layers):
            if layer in all_data[cond] and all_data[cond][layer]:
                steps, norms = zip(*all_data[cond][layer])
                ax.plot(steps, norms, label=layer, color=colors[layer_idx], linewidth=2)

        ax.set_title(cond.replace("_", " ").title())
        ax.set_xlabel("Training Step")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("L2 Norm")
            ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_3d_surface(all_data: Dict[str, Dict[str, List[tuple]]], layer_name: str, save_path: Path):
    """
    Plot a 3D surface: (training_step x collapse_level) -> L2 norm of `layer_name`.
    Uses severity progression (pure -> severe_collapse).
    """
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    valid_conditions = [c for c in conditions if c in all_data and layer_name in all_data[c] and len(all_data[c][layer_name]) > 0]

    if not valid_conditions:
        print(f"No valid data found for 3D plot of {layer_name}")
        return

    # Collect common steps
    # Just take steps from the first valid condition, assuming they match
    sample_data = all_data[valid_conditions[0]][layer_name]
    steps = [x[0] for x in sample_data]

    # Create Z matrix (collapse_idx x step_idx)
    Z = np.zeros((len(valid_conditions), len(steps)))

    for i, cond in enumerate(valid_conditions):
        cond_data = dict(all_data[cond][layer_name]) # step -> norm
        for j, step in enumerate(steps):
            Z[i, j] = cond_data.get(step, np.nan)

    # Interpolate NaNs if any (simplistic row-wise forward fill)
    for i in range(len(valid_conditions)):
        mask = np.isnan(Z[i])
        if np.any(mask):
            Z[i, mask] = np.nanmean(Z[i]) # fallback

    X, Y = np.meshgrid(steps, np.arange(len(valid_conditions)))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Collapse Level')
    ax.set_zlabel(f'{layer_name} L2 Norm')
    ax.set_yticks(np.arange(len(valid_conditions)))
    ax.set_yticklabels([c.replace('_', ' ').title() for c in valid_conditions])

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    ax.view_init(elev=20, azim=-45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="viz_output")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if results_dir.exists():
        print("Collecting weight norm trajectories...")
        all_data = collect_norm_trajectories(results_dir)

        if all_data:
            print("Plotting faceted norms...")
            plot_faceted_norms(all_data, output_dir / "weight_norms_faceted.png")

            print("Plotting 3D surface...")
            # Use token_embed as default
            plot_3d_surface(all_data, "token_embed", output_dir / "weight_norms_3d_token_embed.png")

            # Use output_head as another interesting one
            plot_3d_surface(all_data, "output_head", output_dir / "weight_norms_3d_output_head.png")

            print("Done!")
        else:
            print("No checkpoints found to process.")
    else:
        print(f"Results directory {results_dir} does not exist.")
