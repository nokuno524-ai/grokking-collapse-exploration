"""
Induction Head Analysis for Modular Arithmetic.

Standard induction heads look for [A, B, ..., A] and predict B. In our
sequence-length-2 setup `(a, b)`, there's no sequence to induce. However,
attention heads can act as "copy" or "permutation" heads by attending specifically
to position 0 or position 1.

We define the "prefix-matching score" here as the extent to which attention
focuses heavily on specific input positions based on the tokens. We track
the maximum attention weight allocated to either position 0 or 1, representing
the formation of a deterministic mapping.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic


def extract_attention_weights(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extract attention weights from the 1-layer ModularArithmeticTransformer.

    Args:
        model: The trained ModularArithmeticTransformer.
        inputs: Input tensor of shape (batch, 2).

    Returns:
        Attention weights of shape (batch, n_heads, seq_len, seq_len).
    """
    batch_size = inputs.shape[0]

    tok = model.token_embed(inputs)
    positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    h = tok + pos

    # Run through the transformer layer manually to get attention weights
    layer = model.transformer.layers[0]

    # Layer norm 1
    h_norm = layer.norm1(h)

    # Multi-head attention
    attn_output, attn_weights = layer.self_attn(
        h_norm, h_norm, h_norm,
        need_weights=True,
        average_attn_weights=False,
    )

    return attn_weights


def compute_induction_score(attn_weights: torch.Tensor) -> float:
    """
    Compute an induction-like score.
    Since seq_len=2, we measure how sharply a head attends to a single token.
    A high score means the head almost exclusively attends to one position.

    Args:
        attn_weights: Shape (batch, n_heads, seq_len, seq_len)

    Returns:
        Average max attention weight across batch and sequence (proxy for sharpness).
    """
    # Max attention weight along the key sequence dimension
    # Shape: (batch, n_heads, seq_len)
    max_attn, _ = attn_weights.max(dim=-1)

    # Average across batch and queries
    # Shape: (n_heads,)
    head_scores = max_attn.mean(dim=(0, 2))

    # The score of the model is the max over all heads
    return head_scores.max().item()


def analyze_checkpoint(checkpoint_path: str, test_inputs: torch.Tensor, device: str = 'cpu') -> float:
    """Analyze a single checkpoint and return its induction score."""
    model = ModularArithmeticTransformer().to(device)

    try:
        # standard checkpoint
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # fallback
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    elif "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()

    with torch.no_grad():
        # process in small batches to save memory
        batch_size = 256
        scores = []
        for i in range(0, test_inputs.shape[0], batch_size):
            batch = test_inputs[i:i+batch_size].to(device)
            attn_weights = extract_attention_weights(model, batch)
            score = compute_induction_score(attn_weights)
            scores.append(score)

    return float(np.mean(scores))


def run_induction_analysis(results_dir: str = "results/exp_c_grid"):
    """
    Run induction head analysis across training trajectories for different conditions.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results dir {results_dir} not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup test data (clean)
    config = DatasetConfig(prime=59)
    _, _, test_inputs, _ = generate_modular_arithmetic(config)

    timelines = {}

    # We will pick a few specific configurations for comparison
    # e.g., wd=1.0, noise=0.0 (pure) vs noise=0.15 (collapse)
    target_configs = [
        ("wd1", "noise0", "seed_42"),
        ("wd1", "noise0.15", "seed_42")
    ]

    for wd, noise, seed in target_configs:
        run_dir = results_path / wd / noise / seed
        if not run_dir.exists():
            continue

        print(f"Analyzing {run_dir}...")

        # Load results.json to get grokking step
        grokking_step = None
        results_json = run_dir / "results.json"
        if results_json.exists():
            with open(results_json, "r") as f:
                res = json.load(f)
                grokking_step = res.get("grokking_step")

        # Find checkpoints
        checkpoints = sorted(list(run_dir.glob("checkpoint_*.pt")),
                           key=lambda x: int(x.stem.split('_')[1]))

        steps = []
        scores = []

        for cp in checkpoints:
            step = int(cp.stem.split('_')[1])
            score = analyze_checkpoint(str(cp), test_inputs, device)
            steps.append(step)
            scores.append(score)

        timelines[f"{wd}_{noise}"] = {
            "steps": steps,
            "scores": scores,
            "grokking_step": grokking_step
        }

    # Plotting
    plt.figure(figsize=(10, 6))
    for name, data in timelines.items():
        plt.plot(data["steps"], data["scores"], label=f"{name} induction score", marker='o')
        if data["grokking_step"]:
            plt.axvline(x=data["grokking_step"], linestyle='--', alpha=0.5,
                       label=f"{name} grok step")

    plt.xlabel("Training Step")
    plt.ylabel("Induction Score (Max Attention Sharpness)")
    plt.title("Induction Head Formation Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = Path("analysis/induction_formation.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    run_induction_analysis()
