"""
Neuron-Level Analysis for Modular Arithmetic.

Tracks individual neuron activation patterns in the feedforward network (FFN).
Computes a proxy for polysemanticity based on activation sparsity/variance.
Highly specialized (monosemantic) neurons will have sparse activations.
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


def get_ffn_activations(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extract the post-GELU activations of the FFN layer.

    Args:
        model: Trained ModularArithmeticTransformer
        inputs: Input tensor (batch, 2)

    Returns:
        Activations of shape (batch, 2, d_ff)
    """
    batch_size = inputs.shape[0]

    tok = model.token_embed(inputs)
    positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    h = tok + pos

    layer = model.transformer.layers[0]
    h_norm = layer.norm1(h)
    attn_output, _ = layer.self_attn(h_norm, h_norm, h_norm, need_weights=False)
    h = h + attn_output

    h_norm2 = layer.norm2(h)
    # The inner FFN activation
    inner_acts = F.gelu(layer.linear1(h_norm2))

    return inner_acts


def compute_polysemanticity(acts: torch.Tensor) -> float:
    """
    Compute a proxy for polysemanticity (inverse of sparsity).
    We use the kurtosis of neuron activations across the dataset.
    High kurtosis -> sparse, spiky activations -> monosemantic.
    Low kurtosis -> uniform, dense activations -> polysemantic.

    Returns:
        Average polysemanticity score (lower is more specialized).
    """
    # acts shape: (batch, seq, d_ff)
    # Flatten over batch and seq
    acts_flat = acts.view(-1, acts.shape[-1]) # (batch*seq, d_ff)

    # Compute variance and mean
    mean = acts_flat.mean(dim=0)
    var = acts_flat.var(dim=0)

    # Fourth moment for kurtosis
    diff = acts_flat - mean
    moment4 = (diff ** 4).mean(dim=0)

    # Add epsilon to prevent div by zero
    kurtosis = moment4 / (var ** 2 + 1e-8)

    # Polysemanticity = 1 / kurtosis (higher score = more polysemantic)
    poly = 1.0 / (kurtosis + 1e-8)

    return poly.mean().item()


def analyze_checkpoint(checkpoint_path: str, test_inputs: torch.Tensor, device: str = 'cpu') -> float:
    """Analyze a single checkpoint and return its polysemanticity score."""
    model = ModularArithmeticTransformer().to(device)

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    elif "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()

    with torch.no_grad():
        all_acts = []
        batch_size = 256
        for i in range(0, test_inputs.shape[0], batch_size):
            batch = test_inputs[i:i+batch_size].to(device)
            acts = get_ffn_activations(model, batch)
            all_acts.append(acts)

        full_acts = torch.cat(all_acts, dim=0)
        score = compute_polysemanticity(full_acts)

    return score


def run_neuron_analysis(results_dir: str = "results/exp_c_grid"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results dir {results_dir} not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = DatasetConfig(prime=59)
    _, _, test_inputs, _ = generate_modular_arithmetic(config)

    timelines = {}
    target_configs = [
        ("wd1", "noise0", "seed_42"),
        ("wd1", "noise0.15", "seed_42")
    ]

    for wd, noise, seed in target_configs:
        run_dir = results_path / wd / noise / seed
        if not run_dir.exists():
            continue

        print(f"Analyzing {run_dir}...")

        grokking_step = None
        results_json = run_dir / "results.json"
        if results_json.exists():
            with open(results_json, "r") as f:
                res = json.load(f)
                grokking_step = res.get("grokking_step")

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

    plt.figure(figsize=(10, 6))
    for name, data in timelines.items():
        plt.plot(data["steps"], data["scores"], label=f"{name} polysemanticity", marker='o')
        if data["grokking_step"]:
            plt.axvline(x=data["grokking_step"], linestyle='--', alpha=0.5,
                       label=f"{name} grok step")

    plt.xlabel("Training Step")
    plt.ylabel("Polysemanticity Score (Inverse Kurtosis)")
    plt.title("Neuron Specialization Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = Path("analysis/neuron_polysemanticity.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    run_neuron_analysis()
