import os
import json
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys
import numpy as np
import torch.nn.functional as F

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.model import ModularArithmeticTransformer

def get_attention_weights(model, x):
    """
    Manually extract attention weights since need_weights=False by default in PyTorch.
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]

    with torch.no_grad():
        tok = model.token_embed(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        mha = model.transformer.layers[0].self_attn
        in_proj_weight = mha.in_proj_weight
        in_proj_bias = mha.in_proj_bias

        # Linear projection
        qkv = F.linear(h, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        n_heads = model.n_heads
        head_dim = model.d_model // n_heads

        q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1)

    return attn

def load_checkpoints(result_dir):
    checkpoints = []
    if not os.path.exists(result_dir):
        return checkpoints
    for f in os.listdir(result_dir):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            step = int(f.split("_")[1].split(".")[0])
            checkpoints.append((step, os.path.join(result_dir, f)))
    checkpoints.sort()
    return checkpoints

def generate_animation(condition="pure", output_file="attention_evolution_pure.html"):
    result_dir = f"results/{condition}"
    checkpoints = load_checkpoints(result_dir)

    if not checkpoints:
        print(f"No checkpoints found for {condition}")
        return

    # Need prime from config
    with open(os.path.join(result_dir, "results.json"), "r") as f:
        res = json.load(f)
        config = res["config"]

    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1)
    )

    # Generate some random data to evaluate attention
    torch.manual_seed(42)
    x = torch.randint(0, config.get("prime", 59), (128, 2))

    steps = []
    attn_maps = []

    for step, ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        attn = get_attention_weights(model, x)
        # Average over batch
        attn_mean = attn.mean(dim=0).cpu().numpy()  # (n_heads, 2, 2)
        steps.append(step)
        attn_maps.append(attn_mean)

    n_heads = config.get("n_heads", 4)

    fig, axes = plt.subplots(1, n_heads, figsize=(3 * n_heads, 3))
    if n_heads == 1:
        axes = [axes]

    im_list = []
    for h in range(n_heads):
        im = axes[h].imshow(attn_maps[0][h], vmin=0, vmax=1, cmap="Blues")
        axes[h].set_title(f"Head {h+1}")
        axes[h].set_xticks([0, 1])
        axes[h].set_xticklabels(["Pos 0", "Pos 1"])
        axes[h].set_yticks([0, 1])
        axes[h].set_yticklabels(["Pos 0", "Pos 1"])
        im_list.append(im)

    plt.suptitle(f"Condition: {condition} | Step: {steps[0]}")

    def update(frame_idx):
        for h in range(n_heads):
            im_list[h].set_data(attn_maps[frame_idx][h])
        plt.suptitle(f"Condition: {condition} | Step: {steps[frame_idx]}")
        return im_list

    ani = animation.FuncAnimation(fig, update, frames=len(steps), blit=False, interval=200)
    ani.save(output_file, writer="html")
    plt.close(fig)
    print(f"Saved animation to {output_file}")

if __name__ == "__main__":
    generate_animation("pure", "results/attention_evolution_pure.html")
    generate_animation("medium_collapse", "results/attention_evolution_medium_collapse.html")
