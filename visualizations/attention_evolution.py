import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import argparse
from pathlib import Path
from src.model import ModularArithmeticTransformer
from src.train import load_checkpoint

def plot_attention_evolution(run_dir, save_dir="visualizations/outputs"):
    os.makedirs(save_dir, exist_ok=True)

    # Find all checkpoints
    run_path = Path(run_dir)
    ckpts = sorted(list(run_path.glob("checkpoint_*.pt")), key=lambda x: int(x.stem.split("_")[1]))

    if not ckpts:
        print(f"No checkpoints found in {run_dir}")
        return

    print(f"Found {len(ckpts)} checkpoints for attention evolution.")

    fig, axes = plt.subplots(1, len(ckpts), figsize=(4 * len(ckpts), 4))
    if len(ckpts) == 1:
        axes = [axes]

    for ax, ckpt_path in zip(axes, ckpts):
        step = int(ckpt_path.stem.split("_")[1])
        state = load_checkpoint(ckpt_path)
        config = state["config"]

        model = ModularArithmeticTransformer(
            prime=config["prime"], d_model=config["d_model"],
            n_heads=config["n_heads"], d_ff=config["d_ff"]
        )
        model.load_state_dict(state["model_state"])

        x = torch.tensor([[5, 10]])
        tok = model.token_embed(x)
        positions = torch.arange(2).unsqueeze(0)
        pos = model.pos_embed(positions)
        h = tok + pos

        # Pass directly to self_attn to get weights
        # need_weights=True, average_attn_weights=False (as per memory)
        with torch.no_grad():
            _, attn_weights = model.transformer.layers[0].self_attn(
                h, h, h, need_weights=True, average_attn_weights=False
            )

        # attn_weights is (batch, num_heads, L, S) -> (1, 4, 2, 2)
        # We average across heads for a simple visualization
        attn = attn_weights[0].mean(dim=0).numpy()

        sns.heatmap(attn, cmap="viridis", ax=ax, cbar=False, vmin=0.0, vmax=1.0)
        ax.set_title(f"Step {step}")
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["a (5)", "b (10)"])
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["a (5)", "b (10)"], rotation=0)

    plt.tight_layout()
    out_path = Path(save_dir) / f"{run_path.name}_attention_evolution.png"
    plt.savefig(out_path)
    plt.savefig(out_path.with_suffix('.pdf'))
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    plot_attention_evolution(args.run_dir)
