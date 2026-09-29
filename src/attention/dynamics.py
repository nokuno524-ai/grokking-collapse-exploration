"""
Driver script to analyze and visualize the temporal dynamics of attention patterns
across training trajectories and data conditions.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.model import ModularArithmeticTransformer
from src.data import generate_modular_arithmetic, DatasetConfig
from src.attention.visualizer import (
    plot_attention_grid,
    plot_attention_entropy,
    plot_head_specialization_clusters,
    animate_attention_evolution
)

def extract_attention_patterns(
    model: ModularArithmeticTransformer,
    inputs: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Extract attention patterns from the model for the given inputs.

    Args:
        model: The transformer model
        inputs: Input tensor (batch_size, seq_len)
        device: Device to run on

    Returns:
        Attention tensor of shape (n_layers, n_heads, seq_len, seq_len)
        averaged over the batch.
    """
    model.eval()
    model.to(device)
    inputs = inputs.to(device)

    patterns = []

    # Fast path SDPA does not return attention weights. We must disable it during eval to ensure we can get hooks/weights if needed
    torch.backends.mha.set_fastpath_enabled(False)

    captured_weights = []
    def out_proj_pre_hook(module, args):
        # We can extract the attention from the q, k, v that has already been combined.
        # Alternatively, we can patch MHA to return weights properly.
        # But wait, PyTorch nn.MultiheadAttention provides a way to get attention weights if we call it directly,
        # but within TransformerEncoderLayer we don't control the call.
        # So we hook into `mha.forward` instead to intercept the arguments and enforce `average_attn_weights=False`.
        pass

    # Let's wrap the MHA forward to ensure need_weights=True and average_attn_weights=False
    for layer in model.transformer.layers:
        mha = layer.self_attn
        original_forward = mha.forward

        def wrapped_forward(query, key, value, key_padding_mask=None,
                            need_weights=True, attn_mask=None,
                            average_attn_weights=False, is_causal=False):
            # Enforce returning per-head weights
            out, weights = original_forward(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                average_attn_weights=False,
                is_causal=is_causal
            )
            # PyTorch 2.0+ returns weights as (batch, n_heads, target_len, source_len) when average_attn_weights=False
            captured_weights.append(weights.detach())
            return out, weights

        mha.forward = wrapped_forward

    try:
        with torch.no_grad():
            model(inputs)
    finally:
        # Restore original forward
        for layer in model.transformer.layers:
            # We assume it's just the one layer. If multiple, we'd need to map them back correctly, but we can just do:
            del layer.self_attn.forward
            # (which restores the bound method from the class). Wait, we assigned a function to an instance, so we can just delete it from the instance dictionary.

    # captured_weights has one tensor per layer: (batch, n_heads, seq_len, seq_len)
    avg_patterns = []
    for w in captured_weights:
        # average over batch
        avg_patterns.append(w.mean(dim=0).cpu().numpy())

    return np.stack(avg_patterns) # (n_layers, n_heads, seq_len, seq_len)


def compute_shannon_entropy(attention: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy of attention distributions.

    Args:
        attention: Attention tensor (n_layers, n_heads, seq_len, seq_len)

    Returns:
        Entropy tensor (n_layers, n_heads) - averaged over seq_len
    """
    # Clip to avoid log(0)
    attn = np.clip(attention, 1e-12, 1.0)
    # Entropy per query token: -sum(p * log(p))
    entropy = -np.sum(attn * np.log2(attn), axis=-1)  # (n_layers, n_heads, seq_len)
    # Average over sequence length
    return np.mean(entropy, axis=-1)  # (n_layers, n_heads)


def load_run_history(run_dir: Path) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Load the history and grokking step from a run directory."""
    results_file = run_dir / "results.json"
    if not results_file.exists():
        return [], None

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        return data.get("history", []), data.get("grokking_step")
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return [], None


def analyze_dynamics(
    run_dir: Path,
    output_dir: Path,
    prime: int = 59,
    device_str: str = "cpu"
):
    """
    Analyze attention dynamics for a specific run.
    Extracts patterns at each checkpoint, generates plots.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # Load history to find checkpoints
    history, grok_step = load_run_history(run_dir)
    if not history:
        print(f"No history found in {run_dir}")
        return

    # Get a batch of data to compute attention over
    config = DatasetConfig(prime=prime)
    _, _, test_in, _ = generate_modular_arithmetic(config)
    # Use a small fixed batch to keep it fast and consistent
    eval_batch = test_in[:256]

    steps = []
    entropies = []
    patterns_over_time = []

    # Checkpoints to process
    checkpoint_files = sorted(list(run_dir.glob("checkpoint_*.pt")),
                             key=lambda x: int(x.stem.split('_')[1]))

    if not checkpoint_files:
        print(f"No checkpoints found in {run_dir}")
        return

    print(f"Analyzing {len(checkpoint_files)} checkpoints from {run_dir.name}...")

    # Instantiate an empty model
    # We read config from the first checkpoint
    ckpt = torch.load(checkpoint_files[0], map_location="cpu")
    cfg = ckpt.get("config", {})
    d_model = cfg.get("d_model", 128)
    n_heads = cfg.get("n_heads", 4)

    model = ModularArithmeticTransformer(
        prime=prime, d_model=d_model, n_heads=n_heads
    )

    for ckpt_path in checkpoint_files:
        step = int(ckpt_path.stem.split('_')[1])
        steps.append(step)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

        attn = extract_attention_patterns(model, eval_batch, device)
        patterns_over_time.append(attn)

        ent = compute_shannon_entropy(attn)
        entropies.append(ent)

    steps = np.array(steps)
    patterns_over_time = np.array(patterns_over_time) # (steps, n_layers, n_heads, seq_len, seq_len)
    entropies = np.array(entropies) # (steps, n_layers, n_heads)

    # 1. Plot Attention Grids at key phases (init, pre-grok, post-grok, final)
    key_steps = [steps[0], steps[-1]]
    if grok_step:
        # Find closest steps before and after grokking
        pre_grok = steps[steps < grok_step]
        if len(pre_grok) > 0:
            key_steps.append(pre_grok[-1])
        post_grok = steps[steps >= grok_step]
        if len(post_grok) > 0:
            key_steps.append(post_grok[0])

    key_steps = sorted(list(set(key_steps)))

    for step in key_steps:
        idx = np.where(steps == step)[0][0]
        pattern = patterns_over_time[idx]

        phase = "final"
        if step == steps[0]: phase = "init"
        elif grok_step and step < grok_step: phase = "pre_grok"
        elif grok_step and step >= grok_step: phase = "post_grok"

        plot_attention_grid(
            pattern,
            output_dir / f"attention_grid_step_{step}_{phase}.png",
            title=f"Attention Patterns - Step {step} ({phase})"
        )

    # 2. Plot Entropy over time
    # Flatten layer/head dims for plotting
    flat_entropies = entropies.reshape(len(steps), -1)
    labels = [f"L{l}H{h}" for l in range(entropies.shape[1]) for h in range(entropies.shape[2])]

    plot_attention_entropy(
        steps,
        flat_entropies,
        output_dir / "attention_entropy_dynamics.png",
        labels=labels
    )

    # 3. Head Specialization Clustering (using final checkpoint)
    plot_head_specialization_clusters(
        patterns_over_time[-1],
        output_dir / "head_clustering_final.png",
        title=f"Head Specialization (Final Step {steps[-1]})"
    )

    # 4. Animate evolution of a specific head (e.g., L0H0) or all
    # Just animate head 0 for brevity
    head_history = patterns_over_time[:, 0, 0, :, :] # (steps, seq_len, seq_len)
    animate_attention_evolution(
        head_history,
        output_dir / "attention_evolution_L0H0.gif",
        steps=steps,
        title="L0H0 Evolution"
    )

    print(f"Finished analyzing dynamics. Plots saved to {output_dir}")
    return steps, entropies, patterns_over_time


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--output-dir", type=str, default="analysis/dynamics", help="Output directory")
    args = parser.parse_args()

    analyze_dynamics(Path(args.run_dir), Path(args.output_dir))
