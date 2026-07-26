import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from src.model import ModularArithmeticTransformer

def load_results(directory):
    results = {}

    # 1. Load top-level directories
    for entry in os.scandir(directory):
        if entry.is_dir() and entry.name not in ["grid", "attention", "circuits", "dashboard", "statistics"]:
            res_file = os.path.join(entry.path, "results.json")
            if os.path.exists(res_file):
                with open(res_file, "r") as f:
                    data = json.load(f)
                    results[entry.name] = data

    # 2. Load grid data (multi-seed) to construct a dataframe for accuracy CIs
    grid_data = []
    grid_dir = os.path.join(directory, "grid")
    if os.path.exists(grid_dir):
        for root, dirs, files in os.walk(grid_dir):
            if "results.json" in files:
                with open(os.path.join(root, "results.json"), "r") as f:
                    res = json.load(f)
                    config = res.get("config", {})
                    history = res.get("history", [])

                    collapse_level = config.get("collapse_level", 0.0)
                    seed = config.get("seed", 42)

                    # map level back to named string for easy plotting
                    cond_name = "Unknown"
                    if collapse_level == 0.0: cond_name = "pure"
                    elif collapse_level == 0.25: cond_name = "low_collapse"
                    elif collapse_level == 0.5: cond_name = "medium_collapse"
                    elif collapse_level == 0.75: cond_name = "severe_collapse"
                    elif collapse_level == 1.0: cond_name = "high_collapse"

                    for h in history:
                        grid_data.append({
                            "condition": cond_name,
                            "seed": seed,
                            "step": h["step"],
                            "test_acc": h.get("test_acc", 0.0),
                            "train_loss": h.get("train_loss", float('inf')),
                            "test_loss": h.get("test_loss", float('inf'))
                        })

    # fallback: if no grid exists, synthesize dataframe from top-level runs (single seed)
    if len(grid_data) == 0:
        for condition, data in results.items():
            if "history" in data:
                seed = data.get("config", {}).get("seed", 42)
                for h in data["history"]:
                    grid_data.append({
                        "condition": condition,
                        "seed": seed,
                        "step": h["step"],
                        "test_acc": h.get("test_acc", 0.0),
                        "train_loss": h.get("train_loss", float('inf')),
                        "test_loss": h.get("test_loss", float('inf'))
                    })

    df = pd.DataFrame(grid_data)
    return results, df

def plot_learning_curves_with_ci(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Add training loss lines faintly in the background, test loss solidly
    # Plotting everything on one axis can be messy with CIs, but we'll use lineplot twice
    # First train loss
    sns.lineplot(data=df, x="step", y="train_loss", hue="condition", ax=axes[0], estimator='mean', errorbar=None, alpha=0.3, legend=False)
    # Then test loss
    sns.lineplot(data=df, x="step", y="test_loss", hue="condition", ax=axes[0], estimator='mean', errorbar=('ci', 95))

    axes[0].set_yscale('log')
    axes[0].set_title("Training Loss (Faint) & Validation Loss (Solid, 95% CI)")

    # Accuracy curves with Confidence Intervals
    sns.lineplot(data=df, x="step", y="test_acc", hue="condition", ax=axes[1], estimator='mean', errorbar=('ci', 95))

    # Annotate grokking transition for Pure (where mean crosses 0.9)
    pure_mean = df[df["condition"] == "pure"].groupby("step")["test_acc"].mean()
    grok_step = pure_mean[pure_mean > 0.9].index.min()

    if not pd.isna(grok_step):
        axes[1].axvline(x=grok_step, color='black', linestyle='--', alpha=0.5)
        axes[1].text(grok_step + 500, 0.5, f"Grokking\n(Step {int(grok_step)})", color='black')

    axes[1].set_title("Validation Accuracy (95% CI)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_weight_norms(results, save_path):
    plt.figure(figsize=(8, 5))
    for condition, data in results.items():
        if "history" not in data:
            continue
        history = data["history"]
        if "weight_norm" not in history[0]:
            continue

        steps = [h["step"] for h in history]
        norms = [h["weight_norm"] for h in history]

        plt.plot(steps, norms, label=condition, linewidth=2)

    plt.xlabel("Steps")
    plt.ylabel("L2 Weight Norm")
    plt.title("Weight Norm Evolution During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_maps(model_path, title, save_path):
    if not os.path.exists(model_path):
        return

    model = ModularArithmeticTransformer()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict["model_state"] if "model_state" in state_dict else state_dict)
    model.eval()

    prime = 59
    a = torch.arange(prime)
    x = torch.stack([a, torch.zeros_like(a)], dim=-1)

    with torch.no_grad():
        tok = model.token_embed(x)
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos
        h = model.transformer(h)
        h = model.ln(h)
        reps = h.mean(dim=1).numpy()

    plt.figure(figsize=(10, 6))
    sns.heatmap(reps[:, :32], cmap="viridis")
    plt.title(title)
    plt.xlabel("Hidden Dimension (First 32)")
    plt.ylabel("Input 'a' (where b=0)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tsne_representations(model_path, title, save_path):
    if not os.path.exists(model_path):
        return

    model = ModularArithmeticTransformer()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict["model_state"] if "model_state" in state_dict else state_dict)
    model.eval()

    prime = 59
    a = torch.arange(prime)
    b = torch.arange(prime)
    grid_a, grid_b = torch.meshgrid(a, b, indexing='ij')
    x = torch.stack([grid_a.flatten(), grid_b.flatten()], dim=-1)

    with torch.no_grad():
        tok = model.token_embed(x)
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos
        h = model.transformer(h)
        h = model.ln(h)
        reps = h.mean(dim=1).numpy()
        targets = ((grid_a + grid_b) % prime).flatten().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reps_2d = tsne.fit_transform(reps)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reps_2d[:, 0], reps_2d[:, 1], c=targets, cmap='hsv', alpha=0.6, s=10)
    plt.colorbar(scatter, label='(a + b) mod p')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    os.makedirs("results/dashboard", exist_ok=True)
    results, df = load_results("results")

    if len(df) > 0:
        plot_learning_curves_with_ci(df, "results/dashboard/learning_curves.png")

    plot_weight_norms(results, "results/dashboard/weight_norms.png")

    plot_feature_maps("results/pure/checkpoint_5000.pt", "Feature Maps (Pre-Grokking, Pure)", "results/dashboard/feature_map_pre.png")
    plot_feature_maps("results/pure/checkpoint_50000.pt", "Feature Maps (Post-Grokking, Pure)", "results/dashboard/feature_map_post.png")
    plot_feature_maps("results/high_collapse/checkpoint_50000.pt", "Feature Maps (High Collapse)", "results/dashboard/feature_map_collapse.png")

    plot_tsne_representations("results/pure/checkpoint_5000.pt", "t-SNE Internal Reps (Pre-Grokking, Step 5000)", "results/dashboard/tsne_pre.png")
    plot_tsne_representations("results/pure/checkpoint_50000.pt", "t-SNE Internal Reps (Post-Grokking, Step 50000)", "results/dashboard/tsne_post.png")
    plot_tsne_representations("results/high_collapse/checkpoint_50000.pt", "t-SNE Internal Reps (High Collapse, Step 50000)", "results/dashboard/tsne_collapse.png")

if __name__ == "__main__":
    main()
