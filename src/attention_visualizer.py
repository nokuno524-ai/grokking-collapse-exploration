import argparse
import glob
import json
import os
import re
from pathlib import Path

import torch
import numpy as np

from src.model import ModularArithmeticTransformer
from src.attention.extraction import extract_attention, compute_entropy
from src.attention.visualization import plot_attention_heatmaps, plot_attention_trajectory, plot_critical_token_trajectory


def get_checkpoints(run_dir: str):
    """Get sorted list of checkpoints in a run directory."""
    checkpoints = glob.glob(os.path.join(run_dir, "checkpoint_*.pt"))

    def extract_step(path):
        m = re.search(r'checkpoint_(\d+)\.pt', path)
        return int(m.group(1)) if m else -1

    return sorted(checkpoints, key=extract_step)


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]

    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Visualize attention patterns across checkpoints.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run directory.")
    parser.add_argument("--prompt", type=int, nargs=2, default=[7, 13], help="Two integers representing the prompt (a, b).")
    parser.add_argument("--output-dir", type=str, default="analysis/attention_vis", help="Directory to save visualizations.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist.")
        return

    output_dir = Path(args.output_dir) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = get_checkpoints(str(run_dir))

    if not checkpoints:
        print(f"No checkpoints found in {run_dir}.")
        return

    print(f"Found {len(checkpoints)} checkpoints.")

    # Load results.json to get grokking step
    results_path = run_dir / "results.json"
    grokking_step = None
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
            grokking_step = results.get("grokking_step")

    steps = []
    entropies = []
    attention_to_critical_list = []
    tokens = [str(args.prompt[0]), str(args.prompt[1])]
    critical_token_idx = 0 # Defaulting to the first token as critical target (e.g. 'a' in a+b)

    for ckpt_path in checkpoints:
        step = int(re.search(r'checkpoint_(\d+)\.pt', ckpt_path).group(1))
        steps.append(step)

        print(f"Processing step {step}...")
        model, config = load_model(ckpt_path, device)

        # Prepare input tensor
        x = torch.tensor([args.prompt], dtype=torch.long, device=device)

        # Extract attention
        with extract_attention(model) as extracted:
            with torch.no_grad():
                model(x)

        if not extracted:
            print(f"Warning: No attention extracted for step {step}")
            continue

        # Get layer 0 attention (batch 0)
        attn = extracted[0][0][0]  # shape: (num_heads, seq_len, seq_len)

        # Calculate entropy
        entropy = compute_entropy(attn)
        entropies.append(entropy)

        # Extract attention to critical token
        # attn is shape (num_heads, seq_len, seq_len)
        # We look at the attention FROM token 1 TO token 0 (critical_token_idx)
        attn_to_crit = attn[:, 1, critical_token_idx] # shape: (num_heads,)
        attention_to_critical_list.append(attn_to_crit)

        # Plot heatmap
        heatmap_path = output_dir / f"heatmap_step_{step}.png"
        plot_attention_heatmaps(
            attentions=attn,
            tokens=tokens,
            layer_idx=0,
            save_path=str(heatmap_path),
            title=f"Attention (Prompt {args.prompt}) - Step {step}"
        )

        # Save sidecar JSON
        sidecar = {
            "run_dir": str(run_dir),
            "step": step,
            "prompt": args.prompt,
            "entropy": entropy
        }
        with open(heatmap_path.with_suffix(".json"), "w") as f:
            json.dump(sidecar, f, indent=2)

    # Plot trajectory
    if steps and entropies:
        trajectory_path = output_dir / "entropy_trajectory.png"
        plot_attention_trajectory(
            steps=steps,
            entropies=entropies,
            grokking_step=grokking_step,
            save_path=str(trajectory_path),
            title=f"Attention Entropy Trajectory (Prompt {args.prompt})"
        )

        sidecar = {
            "run_dir": str(run_dir),
            "prompt": args.prompt,
            "grokking_step": grokking_step
        }
        with open(trajectory_path.with_suffix(".json"), "w") as f:
            json.dump(sidecar, f, indent=2)

    if steps and attention_to_critical_list:
        crit_traj_path = output_dir / "critical_token_trajectory.png"
        attn_to_crit_arr = np.array(attention_to_critical_list) # (num_steps, num_heads)

        plot_critical_token_trajectory(
            steps=steps,
            attention_to_critical=attn_to_crit_arr,
            critical_token_idx=critical_token_idx,
            tokens=tokens,
            grokking_step=grokking_step,
            save_path=str(crit_traj_path),
            title=f"Attention to Critical Token Trajectory (Prompt {args.prompt})"
        )

    print(f"Finished generating visualizations in {output_dir}")

if __name__ == "__main__":
    main()
