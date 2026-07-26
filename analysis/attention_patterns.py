import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from src.model import ModularArithmeticTransformer

def extract_attention(model, x):
    """
    Extract attention maps for a single layer transformer.
    """
    batch_size = x.shape[0]

    # Forward pass up to attention
    tok = model.token_embed(x)
    positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    h = tok + pos

    # Q, K, V
    in_proj_weight = model.transformer.layers[0].self_attn.in_proj_weight
    in_proj_bias = model.transformer.layers[0].self_attn.in_proj_bias

    d_model = model.d_model
    n_heads = model.n_heads
    head_dim = d_model // n_heads

    qkv = F.linear(h, in_proj_weight, in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)

    return attn # (batch_size, n_heads, seq_len, seq_len)

def visualize_attention_heatmaps(attn, title, save_path):
    # attn: (n_heads, 2, 2)
    n_heads = attn.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    for i in range(n_heads):
        sns.heatmap(attn[i].numpy(), annot=True, cmap="YlGnBu", ax=axes[i], vmin=0, vmax=1)
        axes[i].set_title(f"Head {i}")
        axes[i].set_xlabel("Key Position")
        axes[i].set_ylabel("Query Position")
        axes[i].set_xticks([0.5, 1.5])
        axes[i].set_xticklabels(["Pos 0", "Pos 1"])
        axes[i].set_yticks([0.5, 1.5])
        axes[i].set_yticklabels(["Pos 0", "Pos 1"])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_head_similarity(attn):
    # attn: (n_heads, seq_len, seq_len)
    n_heads = attn.shape[0]
    flat_attn = attn.reshape(n_heads, -1)
    sim = torch.zeros(n_heads, n_heads)
    for i in range(n_heads):
        for j in range(n_heads):
            # cosine similarity
            sim[i, j] = F.cosine_similarity(flat_attn[i], flat_attn[j], dim=0)
    return sim

def visualize_head_similarity(sim, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(sim.numpy(), annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Head Index")
    plt.ylabel("Head Index")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def attention_rollout(attn):
    # For a 1-layer transformer, rollout is just the attention map + identity, normalized
    n_heads = attn.shape[0]
    # average across heads
    avg_attn = attn.mean(dim=0)
    # add identity matrix for residual connection
    rollout = avg_attn + torch.eye(avg_attn.shape[0])
    # normalize rows
    rollout = rollout / rollout.sum(dim=-1, keepdim=True)
    return rollout

def main():
    os.makedirs("results/attention", exist_ok=True)

    # We will test on specific checkpoints: pre-grokking (5k), grokking onset (15k), post-grokking (50k)
    # Using pure condition
    steps = [5000, 15000, 50000]
    phases = ["pre-grokking", "grokking-onset", "post-grokking"]

    x_test = torch.tensor([[10, 20]]) # A sample input

    for step, phase in zip(steps, phases):
        ckpt_path = f"results/pure/checkpoint_{step}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_path}, not found.")
            continue

        model = ModularArithmeticTransformer()
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict["model_state"] if "model_state" in state_dict else state_dict)
        model.eval()

        with torch.no_grad():
            attn = extract_attention(model, x_test)
            attn_mean = attn.mean(dim=0) # average over batch -> (n_heads, 2, 2)

            visualize_attention_heatmaps(attn_mean, f"Attention Heatmaps (Pure, {phase}, step {step})", f"results/attention/pure_heatmaps_{step}.png")

            sim = compute_head_similarity(attn_mean)
            visualize_head_similarity(sim, f"Head Similarity (Pure, {phase})", f"results/attention/pure_sim_{step}.png")

            rollout = attention_rollout(attn_mean)
            plt.figure(figsize=(4,3))
            sns.heatmap(rollout.numpy(), annot=True, cmap="YlGnBu", vmin=0, vmax=1)
            plt.title(f"Attention Rollout (Pure, {phase})")
            plt.tight_layout()
            plt.savefig(f"results/attention/pure_rollout_{step}.png")
            plt.close()

    # Also compare across collapse levels at post-grokking
    levels = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]
    for level in levels:
        ckpt_path = f"results/{level}/checkpoint_50000.pt"
        if not os.path.exists(ckpt_path):
            continue
        model = ModularArithmeticTransformer()
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict["model_state"] if "model_state" in state_dict else state_dict)
        model.eval()

        with torch.no_grad():
            attn = extract_attention(model, x_test).mean(dim=0)
            visualize_attention_heatmaps(attn, f"Attention Heatmaps ({level}, step 50000)", f"results/attention/{level}_heatmaps_50000.png")

if __name__ == "__main__":
    main()
