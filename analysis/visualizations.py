import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import umap

def load_histories(results_dir="results"):
    histories = {}
    base_path = Path(results_dir)
    for condition_dir in base_path.iterdir():
        if condition_dir.is_dir():
            json_path = condition_dir / "results.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    try:
                        res = json.load(f)
                    except json.JSONDecodeError:
                        continue
                cond_name = res.get("config", {}).get("condition_name", condition_dir.name)
                histories[cond_name] = res.get("history", [])
    return histories

def plot_training_curves(histories, output_path="results/training_curves.png"):
    plt.figure(figsize=(15, 5))

    # Plot Train/Test Acc
    plt.subplot(1, 2, 1)
    for cond, hist in histories.items():
        if not hist: continue
        steps = [h["step"] for h in hist]
        test_acc = [h["test_acc"] for h in hist]
        plt.plot(steps, test_acc, label=cond)
    plt.xlabel("Step")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Trajectories")
    plt.legend()
    plt.grid(True)

    # Plot Train/Test Loss
    plt.subplot(1, 2, 2)
    for cond, hist in histories.items():
        if not hist: continue
        steps = [h["step"] for h in hist]
        test_loss = [h["test_loss"] for h in hist]
        plt.plot(steps, test_loss, label=cond)
    plt.xlabel("Step")
    plt.ylabel("Test Loss (log scale)")
    plt.yscale("log")
    plt.title("Test Loss Trajectories")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_weight_norms(histories, output_path="results/weight_norms.png"):
    plt.figure(figsize=(8, 6))
    for cond, hist in histories.items():
        if not hist: continue
        steps = [h["step"] for h in hist]
        wnorm = [h["weight_norm"] for h in hist]
        plt.plot(steps, wnorm, label=cond)
    plt.xlabel("Step")
    plt.ylabel("Weight Norm")
    plt.title("Total Weight Norm over Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_gradient_flow(histories, results_dir="results", output_path="results/gradient_flow.png"):
    # Since true gradients aren't in histories, compute approximate gradients (W_t - W_{t-1})
    # from checkpoints if possible, else just use differences in weight norms as a very crude proxy.

    plt.figure(figsize=(8, 6))
    for cond, hist in histories.items():
        if not hist: continue
        steps = [h["step"] for h in hist]
        wnorms = [h["weight_norm"] for h in hist]

        # Crude gradient norm proxy based on delta weight norm.
        # (W_t - W_{t-1}) norm would require loading all checkpoints. Let's do delta wnorm here for simplicity.
        grad_proxy = [abs(wnorms[i] - wnorms[i-1]) for i in range(1, len(wnorms))]
        grad_steps = steps[1:]

        # Smooth it
        if len(grad_proxy) > 10:
            window = 10
            grad_proxy = np.convolve(grad_proxy, np.ones(window)/window, mode='valid')
            grad_steps = grad_steps[window-1:]

        plt.plot(grad_steps, grad_proxy, label=cond)

    plt.xlabel("Step")
    plt.ylabel("Approx Gradient Flow (|dW_norm|)")
    plt.title("Gradient Flow Proxy")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_attention_heatmaps(results_dir="results", output_path="results/attention_heatmaps.png"):
    base = Path(results_dir)
    conds = ["pure", "severe_collapse"]
    steps = [5000, 25000, 50000]

    fig, axes = plt.subplots(len(conds), len(steps), figsize=(15, 4 * len(conds)))

    for i, cond in enumerate(conds):
        for j, step in enumerate(steps):
            ax = axes[i, j] if len(conds) > 1 else axes[j]
            ckpt_path = base / cond / f"checkpoint_{step}.pt"
            if not ckpt_path.exists():
                ax.set_title(f"{cond} {step}: missing")
                ax.axis('off')
                continue

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state = ckpt["model_state"]

                # QKV are packed in in_proj_weight for nn.TransformerEncoderLayer
                in_proj = state['transformer.layers.0.self_attn.in_proj_weight']
                d_model = in_proj.shape[1]
                # Q is the first block
                W_q = in_proj[:d_model, :]
                W_k = in_proj[d_model:2*d_model, :]

                # Compute QK^T as a proxy for attention pattern prior to softmax
                # For a random token embed x, att ~ x W_q W_k^T x^T
                # Let's just plot the W_q @ W_k.T matrix (the "attention matrix" core)
                attn_core = (W_q @ W_k.T).numpy()

                sns.heatmap(attn_core[:32, :32], ax=ax, cmap="viridis", cbar=False)
                ax.set_title(f"{cond} @ step {step}")
            except Exception as e:
                ax.set_title(f"Error: {e}")
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_tsne_embeddings(results_dir="results", output_path="results/tsne_embeddings.png"):
    base = Path(results_dir)
    conds = ["pure", "severe_collapse"]
    step = 50000

    plt.figure(figsize=(10, 5))

    for i, cond in enumerate(conds):
        plt.subplot(1, 2, i+1)
        ckpt_path = base / cond / f"checkpoint_{step}.pt"
        if not ckpt_path.exists():
            plt.title(f"{cond}: missing")
            continue

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt["model_state"]
            embeds = state['token_embed.weight'].numpy()

            tsne = TSNE(n_components=2, perplexity=10, random_state=42)
            embeds_2d = tsne.fit_transform(embeds)

            plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], alpha=0.7, s=10)
            plt.title(f"{cond} t-SNE Embeddings")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
        except Exception as e:
            plt.title(f"Error: {e}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def generate_all_visualizations():
    histories = load_histories()
    if not histories:
        print("No histories found to visualize.")
        return

    plot_training_curves(histories)
    plot_weight_norms(histories)
    plot_gradient_flow(histories)
    plot_attention_heatmaps()
    plot_tsne_embeddings()

if __name__ == "__main__":
    generate_all_visualizations()
