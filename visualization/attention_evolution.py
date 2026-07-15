import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import os
import json

from src.model import ModularArithmeticTransformer

def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    model = ModularArithmeticTransformer(
        prime=config["prime"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_layers=config["n_layers"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config

def get_attention_weights(model, inputs):
    # Using the trick: self_attn returns (attn_output, attn_weights) when need_weights=True
    # The ModularArithmeticTransformer has: tok + pos -> transformer -> ...
    tok = model.token_embed(inputs)

    positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
    pos = model.pos_embed(positions)

    x = tok + pos

    # Pass through layer 0
    layer = model.transformer.layers[0]

    # We want per-head weights, average_attn_weights=False gives (batch, n_heads, seq_len, seq_len)
    _, attn_weights = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)

    return attn_weights

def visualize_attention(attn_weights, step, condition, output_dir):
    # attn_weights shape: (batch, n_heads, 2, 2)
    # We'll average over the batch dimension
    avg_attn = attn_weights.mean(dim=0).detach().numpy()
    n_heads = avg_attn.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sns.heatmap(avg_attn[i], annot=True, cmap="YlGnBu", vmin=0, vmax=1, ax=ax,
                    xticklabels=['Pos 0', 'Pos 1'], yticklabels=['Pos 0', 'Pos 1'])
        ax.set_title(f"Head {i+1}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    plt.suptitle(f"Attention Patterns - {condition} (Step {step})")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"attn_{condition}_step_{step}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize attention pattern evolution.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing condition results")
    parser.add_argument("--output-dir", type=str, default="visualization", help="Directory to save visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    conditions = ['pure', 'low_collapse', 'medium_collapse', 'high_collapse', 'severe_collapse']

    # Use a fixed input batch for consistent visualization
    inputs = torch.randint(0, 59, (10, 2))

    for condition in conditions:
        cond_dir = Path(args.results_dir) / condition
        if not cond_dir.exists():
            print(f"Skipping {condition}: directory not found.")
            continue

        # Find checkpoints
        checkpoints = sorted([f for f in os.listdir(cond_dir) if f.startswith("checkpoint_") and f.endswith(".pt")],
                             key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Determine grokking step if available
        results_file = cond_dir / "results.json"
        grokking_step = None
        if results_file.exists():
            with open(results_file, "r") as f:
                res = json.load(f)
                grokking_step = res.get("grokking_step")

        # Select checkpoints (pre-grokking, grokking point, post-grokking)
        selected_ckpts = []
        if checkpoints:
            selected_ckpts.append(checkpoints[0]) # Early
            selected_ckpts.append(checkpoints[len(checkpoints)//2]) # Mid
            selected_ckpts.append(checkpoints[-1]) # Late

        # Also try to include the checkpoint closest to the grokking step
        if grokking_step is not None:
            closest_ckpt = min(checkpoints, key=lambda x: abs(int(x.split('_')[1].split('.')[0]) - grokking_step))
            if closest_ckpt not in selected_ckpts:
                selected_ckpts.append(closest_ckpt)

        # Remove duplicates and sort
        selected_ckpts = sorted(list(set(selected_ckpts)), key=lambda x: int(x.split('_')[1].split('.')[0]))

        for ckpt_name in selected_ckpts:
            step = int(ckpt_name.split('_')[1].split('.')[0])
            ckpt_path = cond_dir / ckpt_name
            model, _ = load_checkpoint(ckpt_path)

            attn_weights = get_attention_weights(model, inputs)
            visualize_attention(attn_weights, step, condition, args.output_dir)
            print(f"Generated visualization for {condition} at step {step}")

if __name__ == "__main__":
    main()
