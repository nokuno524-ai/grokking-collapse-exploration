import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
import os
import glob
import sys
from pathlib import Path

# Configure publication-quality LaTeX fonts
plt.rcParams.update({
    "text.usetex": False, # Use false to avoid requiring local latex installation in sandbox, but we will make it look nice
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Try enabling tex if available
import subprocess
try:
    if subprocess.run(['which', 'latex'], capture_output=True).returncode == 0:
        plt.rcParams.update({"text.usetex": True})
except Exception:
    pass


# Add src to python path so we can import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ModularArithmeticTransformer
from src.config import TrainConfig

def load_checkpoint(ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # Check if 'config' is in checkpoint, else use default
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = TrainConfig(**config_dict)
    else:
        print("Config not found in checkpoint, using default.")
        config = TrainConfig()

    model = ModularArithmeticTransformer(
        prime=config.prime,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers
    )

    # Handle module prefix if it was saved with DDP
    state_dict = checkpoint['model_state']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model, config

def compute_attention_weights(model, inputs):
    """
    Manually compute attention weights since nn.TransformerEncoderLayer
    doesn't return them.
    inputs: (batch, 2)
    """
    with torch.no_grad():
        # 1. Embeddings
        tok = model.token_embed(inputs)
        positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos  # (batch, seq_len=2, d_model)

        # 2. First layer attention projections
        layer = model.transformer.layers[0]
        attn = layer.self_attn

        qkv = F.linear(h, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        batch_size, seq_len, d_model = h.shape
        head_dim = d_model // attn.num_heads

        q = q.view(batch_size, seq_len, attn.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, attn.num_heads, head_dim).transpose(1, 2)

        # Q K^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        return attn_weights

def visualize_attention_heatmap(attn_weights, step, condition, output_dir):
    """
    attn_weights: (batch, num_heads, seq_len, seq_len)
    """
    # Average over batch
    avg_attn = attn_weights.mean(dim=0).numpy() # (num_heads, 2, 2)
    num_heads = avg_attn.shape[0]

    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for i in range(num_heads):
        sns.heatmap(avg_attn[i], ax=axes[i], annot=True, cmap='Blues', vmin=0, vmax=1)
        axes[i].set_title(f'Head {i}')
        axes[i].set_xlabel('Key')
        axes[i].set_ylabel('Query')
        axes[i].set_xticks([0.5, 1.5])
        axes[i].set_xticklabels(['a', 'b'])
        axes[i].set_yticks([0.5, 1.5])
        axes[i].set_yticklabels(['a', 'b'])

    plt.suptitle(f'Attention Patterns - {condition} - Step {step}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'attn_heatmap_{condition}_step{step}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_attention_animation(all_attn_weights, steps, condition, output_dir):
    """
    all_attn_weights: list of (batch, num_heads, seq_len, seq_len) arrays
    """
    if not all_attn_weights:
        return

    num_heads = all_attn_weights[0].shape[1]

    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    def init():
        for ax in axes:
            ax.clear()
        return axes

    def update(frame):
        attn_weights = all_attn_weights[frame]
        step = steps[frame]
        avg_attn = attn_weights.mean(dim=0).numpy() # (num_heads, 2, 2)

        for i in range(num_heads):
            axes[i].clear()
            sns.heatmap(avg_attn[i], ax=axes[i], annot=True, cmap='Blues', vmin=0, vmax=1, cbar=(i == 0), cbar_ax=(None if i > 0 else cbar_ax))
            axes[i].set_title(f'Head {i}')
            axes[i].set_xlabel('Key')
            axes[i].set_ylabel('Query')
            axes[i].set_xticks([0.5, 1.5])
            axes[i].set_xticklabels(['a', 'b'])
            axes[i].set_yticks([0.5, 1.5])
            axes[i].set_yticklabels(['a', 'b'])

        fig.suptitle(f'Attention Patterns - {condition} - Step {step}')
        return axes

    ani = animation.FuncAnimation(fig, update, frames=len(steps), init_func=init, blit=False)

    # Try saving as mp4, fallback to gif
    try:
        ani.save(os.path.join(output_dir, f'attn_animation_{condition}.mp4'), writer='ffmpeg', fps=2, dpi=200)
    except Exception as e:
        print(f"Failed to save mp4: {e}. Saving as gif instead.")
        try:
            ani.save(os.path.join(output_dir, f'attn_animation_{condition}.gif'), writer='pillow', fps=2, dpi=200)
        except Exception as e:
            print(f"Failed to save gif: {e}")

    plt.close()


def compute_entropy(attn_weights):
    """
    Compute Shannon entropy of attention weights.
    attn_weights: (batch, num_heads, seq_len, seq_len)
    Returns scalar
    """
    eps = 1e-10
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1) # (batch, num_heads, seq_len)
    return entropy.mean().item()

def main():
    output_dir = "analysis/attention"
    os.makedirs(output_dir, exist_ok=True)

    # We will look at 'pure' and 'severe_collapse'
    conditions = ["pure", "severe_collapse"]

    # Test inputs
    torch.manual_seed(42)
    test_inputs = torch.randint(0, 59, (100, 2))

    entropy_data = {}

    for condition in conditions:
        entropy_data[condition] = {'steps': [], 'entropies': []}
        all_attn_weights = []

        # Find checkpoints
        ckpt_dir = os.path.join("results", condition)
        # Handle seeds if present
        if os.path.exists(os.path.join(ckpt_dir, "seed_42")):
            ckpt_dir = os.path.join(ckpt_dir, "seed_42")

        ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt"))
        # Sort by step number
        ckpts = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        if not ckpts:
            print(f"No checkpoints found for {condition} at {ckpt_dir}")
            continue

        for i, ckpt_path in enumerate(ckpts):
            step = int(ckpt_path.split('_')[-1].split('.')[0])
            model, _ = load_checkpoint(ckpt_path)

            attn_weights = compute_attention_weights(model, test_inputs)
            all_attn_weights.append(attn_weights)

            # Visualize a few checkpoints
            if i == 0 or i == len(ckpts) // 2 or i == len(ckpts) - 1:
                visualize_attention_heatmap(attn_weights, step, condition, output_dir)

            # Compute entropy
            ent = compute_entropy(attn_weights)
            entropy_data[condition]['steps'].append(step)
            entropy_data[condition]['entropies'].append(ent)

        # Create animation
        create_attention_animation(all_attn_weights, entropy_data[condition]['steps'], condition, output_dir)

    # Plot entropy evolution
    plt.figure(figsize=(10, 6))
    for condition, data in entropy_data.items():
        if data['steps']:
            plt.plot(data['steps'], data['entropies'], label=condition, marker='o')

    plt.title('Attention Entropy Evolution over Training')
    plt.xlabel('Training Steps')
    plt.ylabel('Attention Entropy (nats)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_entropy_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
