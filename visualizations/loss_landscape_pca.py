import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import os
import argparse
from pathlib import Path
from src.train import load_checkpoint

def plot_loss_landscape_pca(run_dir, save_dir="visualizations/outputs"):
    os.makedirs(save_dir, exist_ok=True)
    run_path = Path(run_dir)
    ckpts = sorted(list(run_path.glob("checkpoint_*.pt")), key=lambda x: int(x.stem.split("_")[1]))

    if not ckpts:
        print(f"No checkpoints found in {run_dir}")
        return

    weights = []
    steps = []

    for ckpt_path in ckpts:
        step = int(ckpt_path.stem.split("_")[1])
        state = load_checkpoint(ckpt_path)

        # Flatten all weights into a single vector
        w_vec = []
        for k, v in state["model_state"].items():
            w_vec.append(v.flatten().cpu().numpy())
        import numpy as np
        weights.append(np.concatenate(w_vec))
        steps.append(step)

    weights = np.array(weights)

    if len(weights) < 2:
        print("Need at least 2 checkpoints for PCA")
        return

    pca = PCA(n_components=2)
    coords = pca.fit_transform(weights)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=steps, cmap='viridis', s=100)
    plt.plot(coords[:, 0], coords[:, 1], 'k--', alpha=0.5)

    for i, step in enumerate(steps):
        plt.annotate(f"  {step}", (coords[i, 0], coords[i, 1]))

    plt.colorbar(scatter, label='Training Step')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.title(f"Weight Trajectory PCA: {run_path.name}")

    out_path = Path(save_dir) / f"{run_path.name}_loss_landscape_pca.png"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix('.pdf'))
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    plot_loss_landscape_pca(args.run_dir)
