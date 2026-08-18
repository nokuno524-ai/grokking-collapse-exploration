import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of attention weights for each head.

    Args:
        attention_weights: Tensor of shape (batch_size, n_heads, seq_len, seq_len)

    Returns:
        Tensor of shape (batch_size, n_heads, seq_len) containing entropy per token.
    """
    eps = 1e-9
    probs = attention_weights + eps
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy

def compute_head_specialization(attention_weights: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute how much each head specializes on specific tokens/positions.

    Args:
        attention_weights: (batch, n_heads, seq_len, seq_len)
        targets: Not used for positional specialization but kept for signature consistency if needed.

    Returns:
        Tensor of shape (batch, n_heads) representing the absolute difference
        in attention allocated to position 0 vs position 1.
    """
    avg_attention = attention_weights.mean(dim=2)
    diff = torch.abs(avg_attention[..., 0] - avg_attention[..., 1])
    return diff

def compute_attention_diff(pre_attn: torch.Tensor, post_attn: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute difference between pre and post grokking attention matrices.

    Args:
        pre_attn: Attention weights before grokking (batch, heads, seq, seq)
        post_attn: Attention weights after grokking (batch, heads, seq, seq)

    Returns:
        Tensor of shape (batch, heads) representing the mean absolute difference.
    """
    diff = torch.abs(post_attn - pre_attn)
    return diff.mean(dim=(2, 3))


def extract_attention(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Manually extract multi-head attention weights from ModularArithmeticTransformer.
    The Q, K, V projections must be manually reconstructed from transformer.layers.0.self_attn.in_proj_weight,
    as the default PyTorch TransformerEncoderLayer hardcodes need_weights=False.

    Args:
        model: ModularArithmeticTransformer instance
        inputs: Input tensor (batch, seq_len)

    Returns:
        attention_weights: (batch, n_heads, seq_len, seq_len)
    """
    # Create input representation (tok + pos)
    tok = model.token_embed(inputs)
    pos = model.pos_embed(torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1))
    h = tok + pos

    layer = model.transformer.layers[0]
    in_proj_weight = layer.self_attn.in_proj_weight
    in_proj_bias = layer.self_attn.in_proj_bias

    qkv = torch.nn.functional.linear(h, in_proj_weight, in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    batch_size, seq_len, _ = h.shape
    n_heads = model.n_heads
    d_head = model.d_model // n_heads

    q = q.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    return torch.softmax(attn_scores, dim=-1)

def load_checkpoint(ckpt_path: Path) -> torch.nn.Module:
    """
    Load a model from a checkpoint.

    Args:
        ckpt_path: Path to the checkpoint file

    Returns:
        Loaded ModularArithmeticTransformer model
    """
    # Important: weights_only=False and clean 'module.' prefixes if they exist
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.model import ModularArithmeticTransformer

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = ModularArithmeticTransformer()
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model

def plot_attention_stats(entropy: np.ndarray, specialization: np.ndarray, output_path: Path):
    """
    Generate plots for attention entropy and specialization.
    Args:
        entropy: np.ndarray of shape (n_heads, seq_len)
        specialization: np.ndarray of shape (n_heads,)
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    n_heads = entropy.shape[0]

    im = ax1.imshow(entropy, cmap='viridis', aspect='auto')
    ax1.set_title("Attention Entropy per Head/Position")
    ax1.set_xlabel("Sequence Position")
    ax1.set_ylabel("Head Index")
    fig.colorbar(im, ax=ax1)

    ax2.bar(range(n_heads), specialization)
    ax2.set_title("Head Specialization (Pos 0 vs Pos 1)")
    ax2.set_xlabel("Head Index")
    ax2.set_ylabel("Absolute Difference")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_attention_summaries(entropy: np.ndarray, specialization: np.ndarray, diff: Optional[np.ndarray], output_dir: Path):
    """
    Save aggregated metrics to a JSON file to prevent dumping massive raw tensors.

    Args:
        entropy: np.ndarray of shape (n_heads, seq_len)
        specialization: np.ndarray of shape (n_heads,)
        diff: Optional array of differences
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mean_entropy": float(np.mean(entropy)),
        "entropy_per_head": entropy.mean(axis=1).tolist(),
        "specialization_per_head": specialization.tolist(),
    }
    if diff is not None:
        summary["mean_diff_per_head"] = diff.tolist()

    with open(output_dir / "attention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def main():
    """Main function to run attention visualization on checkpoints."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-ckpt", type=str, help="Path to pre-grokking checkpoint")
    parser.add_argument("--post-ckpt", type=str, help="Path to post-grokking checkpoint")
    parser.add_argument("--output-dir", type=str, default="analysis/attention")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate some random standard input data for testing attention
    # All possible pairs
    p = 59
    inputs = torch.tensor([[a, b] for a in range(p) for b in range(p)], dtype=torch.long)

    pre_attn = None
    post_attn = None
    diff = None

    if args.pre_ckpt and Path(args.pre_ckpt).exists():
        pre_model = load_checkpoint(Path(args.pre_ckpt))
        with torch.no_grad():
            pre_attn = extract_attention(pre_model, inputs)

    if args.post_ckpt and Path(args.post_ckpt).exists():
        post_model = load_checkpoint(Path(args.post_ckpt))
        with torch.no_grad():
            post_attn = extract_attention(post_model, inputs)

    # If we don't have real checkpoints (like in CI or early dev), we create dummy models
    if pre_attn is None and post_attn is None:
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from src.model import ModularArithmeticTransformer
        print("No valid checkpoints provided, using a freshly initialized model for testing.")
        dummy = ModularArithmeticTransformer()
        with torch.no_grad():
            post_attn = extract_attention(dummy, inputs)

    # Process whichever attention matrix we have (preferably post, else pre)
    active_attn = post_attn if post_attn is not None else pre_attn

    if active_attn is not None:
        entropy = compute_attention_entropy(active_attn).mean(dim=0).numpy() # avg over batch
        specialization = compute_head_specialization(active_attn).mean(dim=0).numpy()

        if pre_attn is not None and post_attn is not None:
            diff = compute_attention_diff(pre_attn, post_attn).mean(dim=0).numpy()

        plot_attention_stats(entropy, specialization, output_dir / "attention_stats.png")
        save_attention_summaries(entropy, specialization, diff, output_dir)
        print(f"Generated attention files in {output_dir}")

if __name__ == "__main__":
    main()
