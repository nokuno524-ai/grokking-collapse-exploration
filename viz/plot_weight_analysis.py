import os
import torch
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import re

def compute_weight_norm(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

    total_norm_sq = 0.0
    for key, tensor in state_dict.items():
        # Strip module prefix if exists
        key = key.replace('module.', '')
        total_norm_sq += tensor.float().pow(2).sum().item()

    return total_norm_sq ** 0.5

def plot_weight_analysis(results_dir="results", output_dir="viz/output"):
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    conditions = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

    plt.figure(figsize=(15, 6))

    # Subplot 1: Trajectories
    plt.subplot(1, 2, 1)

    final_norms = {}

    for condition in conditions:
        condition_dir = results_path / condition
        if not condition_dir.exists():
            continue

        checkpoints = list(condition_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            continue

        steps_and_norms = []
        for cp in checkpoints:
            match = re.search(r"checkpoint_(\d+).pt", cp.name)
            if match:
                step = int(match.group(1))
                norm = compute_weight_norm(cp)
                steps_and_norms.append((step, norm))

        steps_and_norms.sort(key=lambda x: x[0])
        steps = [x[0] for x in steps_and_norms]
        norms = [x[1] for x in steps_and_norms]

        plt.plot(steps, norms, label=condition, marker='o')
        if norms:
            final_norms[condition] = norms[-1]

    plt.title("Weight Norm Trajectory")
    plt.xlabel("Step")
    plt.ylabel("Total L2 Norm")
    plt.legend()
    plt.grid(True)

    # Subplot 2: Weight distribution histograms
    plt.subplot(1, 2, 2)

    for condition in conditions:
        condition_dir = results_path / condition
        if not condition_dir.exists():
            continue

        checkpoints = list(condition_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            continue

        # Get the latest checkpoint
        latest_cp = sorted(checkpoints, key=lambda x: int(re.search(r"checkpoint_(\d+).pt", x.name).group(1)))[-1]
        checkpoint = torch.load(latest_cp, map_location='cpu')

        state_dict = checkpoint
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

        all_weights = []
        for key, tensor in state_dict.items():
            if 'weight' in key:
                all_weights.append(tensor.flatten().float())

        if all_weights:
            all_weights_cat = torch.cat(all_weights)
            # Create a histogram using matplotlib
            plt.hist(all_weights_cat.numpy(), bins=50, alpha=0.5, label=condition, density=True)

    plt.title("Final Weight Distributions")
    plt.xlabel("Weight Value")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(out_path / "weight_analysis.png")
    plt.close()

if __name__ == "__main__":
    plot_weight_analysis()
