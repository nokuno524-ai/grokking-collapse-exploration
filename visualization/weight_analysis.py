import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple

def extract_weights(ckpt_path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Extract weight matrices from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    return ckpt['model_state']

def compute_weight_norm(weights: Dict[str, torch.Tensor]) -> float:
    """Compute total L2 norm of all weights."""
    total_norm = 0.0
    for name, w in weights.items():
        if 'weight' in name:
            total_norm += w.norm().item() ** 2
    return total_norm ** 0.5

def extract_singular_values(weights: Dict[str, torch.Tensor], target_layer: str) -> np.ndarray:
    """Extract singular values for a specific 2D weight matrix."""
    w = weights[target_layer]
    # For in_proj_weight which is 3*d_model x d_model
    if len(w.shape) > 2:
        w = w.view(w.size(0), -1)
    elif len(w.shape) < 2:
        return np.array([])

    s = torch.linalg.svdvals(w)
    return s.cpu().numpy()

def analyze_weight_norms_trajectory(base_dir: str = "results", output_dir: str = "results/analysis_output"):
    """Plot weight norm trajectories for all collapse levels."""
    conditions = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

    plt.figure(figsize=(10, 6))

    for condition in conditions:
        cond_dir = os.path.join(base_dir, condition)
        if not os.path.exists(cond_dir):
            continue

        ckpts = glob.glob(os.path.join(cond_dir, "checkpoint_*.pt"))
        if not ckpts:
            continue

        steps = []
        norms = []

        for f in ckpts:
            step = int(f.split("checkpoint_")[1].split(".pt")[0])
            weights = extract_weights(f)
            norm = compute_weight_norm(weights)

            steps.append(step)
            norms.append(norm)

        # Sort by step
        sorted_indices = np.argsort(steps)
        steps = np.array(steps)[sorted_indices]
        norms = np.array(norms)[sorted_indices]

        plt.plot(steps, norms, marker='o', label=condition)

    plt.xlabel("Training Step")
    plt.ylabel("Total Weight L2 Norm")
    plt.title("Weight Norm Evolution Across Collapse Levels")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "weight_norm_trajectories.png"), dpi=150)
    plt.close()

def analyze_weight_distributions(base_dir: str = "results", condition: str = "pure", output_dir: str = "results/analysis_output"):
    """Plot KDE of weight distributions at different training steps."""
    cond_dir = os.path.join(base_dir, condition)
    if not os.path.exists(cond_dir):
        return

    ckpts = glob.glob(os.path.join(cond_dir, "checkpoint_*.pt"))
    if not ckpts:
        return

    steps = sorted([int(f.split("checkpoint_")[1].split(".pt")[0]) for f in ckpts])

    # Pick a few key steps (e.g. beginning, middle, end)
    if len(steps) >= 3:
        key_steps = [steps[0], steps[len(steps)//2], steps[-1]]
    else:
        key_steps = steps

    target_layers = ['token_embed.weight', 'output_head.weight']

    for layer in target_layers:
        plt.figure(figsize=(10, 6))

        for step in key_steps:
            ckpt_path = os.path.join(cond_dir, f"checkpoint_{step}.pt")
            weights = extract_weights(ckpt_path)

            if layer in weights:
                w_flat = weights[layer].flatten().cpu().numpy()
                sns.kdeplot(w_flat, label=f"Step {step}")

        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.title(f"Weight Distribution for {layer} ({condition})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"weight_dist_{condition}_{layer.split('.')[0]}.png"), dpi=150)
        plt.close()

def analyze_singular_values(base_dir: str = "results", condition: str = "pure", output_dir: str = "results/analysis_output"):
    """Plot singular value spectrum evolution for key matrices."""
    cond_dir = os.path.join(base_dir, condition)
    if not os.path.exists(cond_dir):
        return

    ckpts = glob.glob(os.path.join(cond_dir, "checkpoint_*.pt"))
    if not ckpts:
        return

    steps = sorted([int(f.split("checkpoint_")[1].split(".pt")[0]) for f in ckpts])

    if len(steps) >= 3:
        key_steps = [steps[0], steps[len(steps)//2], steps[-1]]
    else:
        key_steps = steps

    # Analyze in_proj_weight which contains Q, K, V
    target_layer = 'transformer.layers.0.self_attn.in_proj_weight'

    plt.figure(figsize=(10, 6))

    for step in key_steps:
        ckpt_path = os.path.join(cond_dir, f"checkpoint_{step}.pt")
        weights = extract_weights(ckpt_path)

        if target_layer in weights:
            s = extract_singular_values(weights, target_layer)
            plt.plot(np.arange(1, len(s)+1), s, marker='o', markersize=3, label=f"Step {step}")

    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value Magnitude")
    plt.title(f"Singular Value Spectrum of Attention Projections ({condition})")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"svd_spectrum_{condition}_attn_proj.png"), dpi=150)
    plt.close()

def run_all_weight_analysis():
    """Run all weight analysis tasks."""
    output_dir = "results/analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    print("Analyzing weight norm trajectories...")
    analyze_weight_norms_trajectory(output_dir=output_dir)

    conditions = ["pure", "low_collapse", "severe_collapse"]
    for cond in conditions:
        print(f"Analyzing weight distributions for {cond}...")
        analyze_weight_distributions(condition=cond, output_dir=output_dir)
        print(f"Analyzing singular values for {cond}...")
        analyze_singular_values(condition=cond, output_dir=output_dir)

if __name__ == "__main__":
    run_all_weight_analysis()
