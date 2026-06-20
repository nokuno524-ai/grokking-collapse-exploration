"""
Circuit Complexity Analysis for Modular Arithmetic.

Measures circuit size (important edge count) and density over training checkpoints.
We use a simple activation attribution proxy: magnitude of the weight matrices
scaled by the mean activation norm. We track this metric for token embedding ->
attention heads, and attention heads -> output logits.
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


def extract_activations(model: torch.nn.Module, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Extract intermediate activations from the model.
    """
    batch_size = inputs.shape[0]
    acts = {}

    tok = model.token_embed(inputs)
    positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    h = tok + pos
    acts['resid_pre'] = h

    layer = model.transformer.layers[0]
    h_norm = layer.norm1(h)
    acts['resid_pre_norm'] = h_norm

    # We manually compute Q, K, V to get head-specific activations
    q = layer.self_attn.in_proj_weight[:model.d_model, :] @ h_norm.transpose(1, 2)
    k = layer.self_attn.in_proj_weight[model.d_model:2*model.d_model, :] @ h_norm.transpose(1, 2)
    v = layer.self_attn.in_proj_weight[2*model.d_model:, :] @ h_norm.transpose(1, 2)

    q = q.transpose(1, 2).view(batch_size, 2, model.n_heads, model.d_model // model.n_heads)
    k = k.transpose(1, 2).view(batch_size, 2, model.n_heads, model.d_model // model.n_heads)
    v = v.transpose(1, 2).view(batch_size, 2, model.n_heads, model.d_model // model.n_heads)

    acts['q'] = q
    acts['k'] = k
    acts['v'] = v

    # Run rest of model
    attn_output, _ = layer.self_attn(h_norm, h_norm, h_norm, need_weights=False)
    h = h + attn_output
    acts['resid_mid'] = h

    h_norm2 = layer.norm2(h)
    ffn_output = layer.linear2(F.gelu(layer.linear1(h_norm2)))
    h = h + ffn_output
    acts['resid_post'] = h

    return acts


def compute_circuit_density(model: torch.nn.Module, acts: Dict[str, torch.Tensor]) -> float:
    """
    Compute a simple proxy for circuit density.
    We measure the participation ratio (L1 norm / L2 norm) of the
    head outputs to measure how sparse/dense the computation is.
    """
    # Use v values as a proxy for head activity
    v = acts['v'] # (batch, seq, n_heads, d_head)
    v_norm = v.norm(dim=-1).mean(dim=(0, 1)) # (n_heads,)

    # Calculate participation ratio: (sum |x|)^2 / sum(x^2)
    # Scaled to be between 1 (sparse) and n_heads (dense)
    l1 = v_norm.sum()
    l2_sq = (v_norm ** 2).sum()

    if l2_sq == 0:
        return 0.0

    pr = (l1 ** 2) / l2_sq
    return pr.item()


def analyze_checkpoint(checkpoint_path: str, test_inputs: torch.Tensor, device: str = 'cpu') -> float:
    """Analyze a single checkpoint and return its circuit density score."""
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
        batch_size = 256
        densities = []
        for i in range(0, test_inputs.shape[0], batch_size):
            batch = test_inputs[i:i+batch_size].to(device)
            acts = extract_activations(model, batch)
            density = compute_circuit_density(model, acts)
            densities.append(density)

    return float(np.mean(densities))


def run_circuit_analysis(results_dir: str = "results/exp_c_grid"):
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
        densities = []

        for cp in checkpoints:
            step = int(cp.stem.split('_')[1])
            density = analyze_checkpoint(str(cp), test_inputs, device)
            steps.append(step)
            densities.append(density)

        timelines[f"{wd}_{noise}"] = {
            "steps": steps,
            "densities": densities,
            "grokking_step": grokking_step
        }

    plt.figure(figsize=(10, 6))
    for name, data in timelines.items():
        plt.plot(data["steps"], data["densities"], label=f"{name} density", marker='o')
        if data["grokking_step"]:
            plt.axvline(x=data["grokking_step"], linestyle='--', alpha=0.5,
                       label=f"{name} grok step")

    plt.xlabel("Training Step")
    plt.ylabel("Circuit Density (Head Participation Ratio)")
    plt.title("Circuit Complexity Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = Path("analysis/circuit_complexity.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    run_circuit_analysis()
