"""
Information Flow Analysis for Modular Arithmetic.

Computes mutual information between intermediate hidden representations
(e.g., post-embedding, post-attention, post-FFN) and the labels.
Since exact MI is intractable for continuous representations, we use a simple
linear probing accuracy proxy (which bounds MI via Fano's inequality) or
variance explanation. Here we fit simple linear probes to measure the
amount of linearly accessible information about the target.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression

try:
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.model import ModularArithmeticTransformer
    from src.data import DatasetConfig, generate_modular_arithmetic


def get_representations(model: torch.nn.Module, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Extract representations from different layers."""
    batch_size = inputs.shape[0]
    reps = {}

    tok = model.token_embed(inputs)
    positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    h = tok + pos
    reps['embedding'] = h.mean(dim=1)  # Pool across pos

    layer = model.transformer.layers[0]
    h_norm = layer.norm1(h)
    attn_output, _ = layer.self_attn(h_norm, h_norm, h_norm, need_weights=False)
    h = h + attn_output
    reps['post_attn'] = h.mean(dim=1)

    h_norm2 = layer.norm2(h)
    ffn_output = layer.linear2(F.gelu(layer.linear1(h_norm2)))
    h = h + ffn_output
    reps['post_ffn'] = h.mean(dim=1)

    return reps


def compute_mi_proxy(reps: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute a proxy for Mutual Information.
    We fit a simple Logistic Regression classifier and use its accuracy.
    """
    X = reps.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()

    # In scikit-learn >= 1.3, multi_class='multinomial' is the default and the arg is deprecated
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')

    try:
        clf.fit(X, y)
        acc = clf.score(X, y)
        return acc
    except Exception:
        # e.g. if the representations blow up to NaN
        return 0.0


def analyze_checkpoint(checkpoint_path: str, inputs: torch.Tensor, targets: torch.Tensor, device: str = 'cpu') -> Dict[str, float]:
    """Analyze a single checkpoint and return MI proxy scores for each layer."""
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
        all_reps = {'embedding': [], 'post_attn': [], 'post_ffn': []}
        batch_size = 256

        for i in range(0, inputs.shape[0], batch_size):
            batch = inputs[i:i+batch_size].to(device)
            reps = get_representations(model, batch)

            for k, v in reps.items():
                all_reps[k].append(v)

        # Concatenate and compute MI
        scores = {}
        for k in all_reps:
            full_rep = torch.cat(all_reps[k], dim=0)
            scores[k] = compute_mi_proxy(full_rep, targets)

    return scores


def run_info_flow_analysis(results_dir: str = "results/exp_c_grid"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results dir {results_dir} not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = DatasetConfig(prime=59)
    # Use full dataset for stable probing
    _, _, test_inputs, test_targets = generate_modular_arithmetic(config)

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
        scores_emb = []
        scores_attn = []
        scores_ffn = []

        for cp in checkpoints:
            step = int(cp.stem.split('_')[1])
            scores = analyze_checkpoint(str(cp), test_inputs, test_targets, device)

            steps.append(step)
            scores_emb.append(scores['embedding'])
            scores_attn.append(scores['post_attn'])
            scores_ffn.append(scores['post_ffn'])

        timelines[f"{wd}_{noise}"] = {
            "steps": steps,
            "embedding": scores_emb,
            "post_attn": scores_attn,
            "post_ffn": scores_ffn,
            "grokking_step": grokking_step
        }

    # Create subplots for each layer
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layers = ["embedding", "post_attn", "post_ffn"]

    for i, layer in enumerate(layers):
        ax = axes[i]
        for name, data in timelines.items():
            ax.plot(data["steps"], data[layer], label=f"{name}", marker='o')
            if data["grokking_step"]:
                ax.axvline(x=data["grokking_step"], linestyle='--', alpha=0.5)

        ax.set_title(f"{layer} Information Flow")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Linear Probe Accuracy (MI Proxy)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    out_path = Path("analysis/information_flow.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    run_info_flow_analysis()
