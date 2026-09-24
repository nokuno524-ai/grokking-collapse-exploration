"""
Attention pattern visualization suite.

This script loads a checkpoint (or sweep of checkpoints), extracts attention maps,
and renders:
(a) per-head heatmap grids across training time
(b) head-similarity-over-time matrices
(c) aggregate attention-entropy trajectories per head

Outputs are PNG figures and a JSON of underlying numbers.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import model and extraction tools
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ModularArithmeticTransformer
from src.attention_extraction import extract_attention, compute_entropy

def get_checkpoints(directory: Path) -> List[Tuple[int, Path]]:
    """
    Auto-discover checkpoint_*.pt files in a directory and sort by step.

    Args:
        directory: The directory containing checkpoint files.

    Returns:
        List of tuples (step_number, file_path) sorted by step_number.
    """
    checkpoints = []
    ckpt_re = re.compile(r"^checkpoint_(\d+)\.pt$")

    for p in directory.glob("checkpoint_*.pt"):
        m = ckpt_re.match(p.name)
        if m:
            step = int(m.group(1))
            checkpoints.append((step, p))

    return sorted(checkpoints, key=lambda x: x[0])


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[ModularArithmeticTransformer, Dict[str, Any], int]:
    """
    Load a model and its configuration from a checkpoint file.

    Args:
        ckpt_path: Path to the checkpoint file.
        device: Torch device to load the model onto.

    Returns:
        Tuple of (model, config_dict, step).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    step = ckpt.get("step", 0)

    model = ModularArithmeticTransformer(
        prime=cfg.get("prime", 59),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        d_ff=cfg.get("d_ff", 512),
        n_layers=cfg.get("n_layers", 1),
    )

    # Load state dict
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, cfg, step


def main():
    parser = argparse.ArgumentParser(description="Visualize attention patterns from checkpoints.")
    parser.add_argument("--checkpoint", type=Path, help="Path to a single checkpoint file.")
    parser.add_argument("--directory", type=Path, help="Path to a directory containing checkpoint_*.pt files.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/attention"), help="Directory to save outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input generation.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to evaluate attention on.")

    args = parser.parse_args()

    if not args.checkpoint and not args.directory:
        parser.error("Either --checkpoint or --directory must be provided.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = []
    if args.checkpoint:
        if not args.checkpoint.exists():
            print(f"Error: Checkpoint {args.checkpoint} not found.")
            sys.exit(1)
        checkpoints.append((0, args.checkpoint))  # Fake step 0 for single checkpoint
    elif args.directory:
        if not args.directory.exists():
            print(f"Error: Directory {args.directory} not found.")
            sys.exit(1)
        checkpoints = get_checkpoints(args.directory)
        if not checkpoints:
            print(f"Error: No checkpoint_*.pt files found in {args.directory}.")
            sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints to process.")

    all_attns = []

    for step, ckpt_path in checkpoints:
        print(f"Processing checkpoint at step {step}...")
        model, cfg, _ = load_model_from_checkpoint(ckpt_path, device)

        # Generate dummy input data: [batch_size, 2]
        # For simplicity, we just random sample numbers between 0 and prime-1
        prime = cfg.get("prime", 59)
        torch.manual_seed(args.seed)
        inputs = torch.randint(0, prime, (args.num_samples, 2), device=device)

        with extract_attention(model) as extracted_maps:
            with torch.no_grad():
                _ = model(inputs)

        if extracted_maps:
            # We only have 1 layer, so we take the first attention map
            # extracted_maps[0] has shape [batch_size, num_heads, seq_len, seq_len]
            attns = extracted_maps[0]
            all_attns.append((step, attns))

            # (a) per-head heatmap grids
            render_attention_heatmaps(attns, step, args.output_dir)

    if all_attns:
        # (b) head-similarity-over-time matrices
        render_head_similarity(all_attns, args.output_dir)

        # (c) aggregate attention-entropy trajectories per head
        render_attention_entropy(all_attns, args.output_dir)
        print(f"Done! Results saved to {args.output_dir}")
    else:
        print("No attention maps were extracted. Make sure the model has multihead attention layers.")

if __name__ == "__main__":
    main()

def render_attention_heatmaps(attns: np.ndarray, step: int, output_dir: Path):
    """
    Render per-head heatmap grids for a single checkpoint.

    Args:
        attns: Attention maps array of shape [batch_size, num_heads, seq_len, seq_len].
        step: Checkpoint step.
        output_dir: Directory to save the PNG file.
    """
    if not HAS_MATPLOTLIB:
        return

    num_heads = attns.shape[1]

    # Average across the batch dimension
    mean_attns = attns.mean(axis=0)  # [num_heads, seq_len, seq_len]

    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(mean_attns[h], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Head {h}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        # Ensure integer ticks for short sequences
        ax.set_xticks(range(mean_attns.shape[-1]))
        ax.set_yticks(range(mean_attns.shape[-2]))

    fig.colorbar(im, ax=axes, shrink=0.6, label="Attention Weight")
    fig.suptitle(f"Attention Heatmaps (Step {step})")

    plt.savefig(output_dir / f"attention_heatmaps_step_{step}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

def render_head_similarity(all_attns: List[Tuple[int, np.ndarray]], output_dir: Path):
    """
    Render cosine similarity of attention distributions between checkpoints over time.

    Args:
        all_attns: List of tuples (step, attns_array).
        output_dir: Directory to save the PNG file.
    """
    if not HAS_MATPLOTLIB or len(all_attns) < 2:
        return

    steps = [x[0] for x in all_attns]
    num_heads = all_attns[0][1].shape[1]
    num_steps = len(steps)

    # Pre-compute flattened mean attention maps
    flat_maps = []
    for _, attns in all_attns:
        # Average over batch
        mean_attns = attns.mean(axis=0)  # [num_heads, seq_len, seq_len]
        flat_maps.append(mean_attns.reshape(num_heads, -1))

    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        sim_matrix = np.zeros((num_steps, num_steps))

        for i in range(num_steps):
            v_i = flat_maps[i][h]
            norm_i = np.linalg.norm(v_i)
            for j in range(num_steps):
                v_j = flat_maps[j][h]
                norm_j = np.linalg.norm(v_j)

                if norm_i > 0 and norm_j > 0:
                    sim_matrix[i, j] = np.dot(v_i, v_j) / (norm_i * norm_j)
                else:
                    sim_matrix[i, j] = 0.0

        ax = axes[h]
        im = ax.imshow(sim_matrix, cmap="plasma", vmin=0, vmax=1)
        ax.set_title(f"Head {h} Similarity")
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Step Index")

        # Add a few ticks
        num_ticks = min(5, num_steps)
        tick_indices = np.linspace(0, num_steps - 1, num_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)
        ax.set_xticklabels([steps[i] for i in tick_indices])
        ax.set_yticklabels([steps[i] for i in tick_indices])

    fig.colorbar(im, ax=axes, shrink=0.6, label="Cosine Similarity")
    fig.suptitle("Head Similarity Over Time")

    plt.savefig(output_dir / "head_similarity_over_time.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

def render_attention_entropy(all_attns: List[Tuple[int, np.ndarray]], output_dir: Path):
    """
    Render aggregate attention-entropy trajectories per head over time.

    Args:
        all_attns: List of tuples (step, attns_array).
        output_dir: Directory to save the PNG file and JSON data.
    """
    if not all_attns:
        return

    steps = [x[0] for x in all_attns]
    num_heads = all_attns[0][1].shape[1]

    entropy_data = {f"head_{h}": [] for h in range(num_heads)}
    entropy_data["steps"] = steps

    for _, attns in all_attns:
        # Calculate mean entropy over batch and sequence positions for each head
        batch_size, n_heads, seq_len, _ = attns.shape
        mean_entropies = np.zeros(n_heads)

        for h in range(n_heads):
            head_entropies = []
            for b in range(batch_size):
                for q in range(seq_len):
                    w = attns[b, h, q, :]
                    e = compute_entropy(w)
                    head_entropies.append(e)

            mean_e = np.mean(head_entropies)
            mean_entropies[h] = float(mean_e)
            entropy_data[f"head_{h}"].append(float(mean_e))

    # Save to JSON
    with open(output_dir / "attention_entropy.json", "w") as f:
        json.dump(entropy_data, f, indent=2)

    if not HAS_MATPLOTLIB:
        return

    # Plot
    plt.figure(figsize=(10, 6))
    for h in range(num_heads):
        plt.plot(steps, entropy_data[f"head_{h}"], label=f"Head {h}", marker='o', markersize=3)

    plt.title("Attention Entropy per Head Over Time")
    plt.xlabel("Training Step")
    plt.ylabel("Entropy (nats)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(output_dir / "attention_entropy_over_time.png", bbox_inches="tight", dpi=150)
    plt.close()

# Update main to use the rendering functions
