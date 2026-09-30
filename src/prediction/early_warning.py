"""
Early-warning indicators for forecasting grokking.
"""

import numpy as np
import torch
import json
from typing import List, Dict, Optional, Tuple
import sys
import os

# Ensure src is importable for models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def compute_effective_rank(singular_values: np.ndarray) -> float:
    """
    Compute effective rank based on singular values.
    Effective rank is exp(-sum p log p) where p are normalized singular values.

    Args:
        singular_values (np.ndarray): Array of singular values.

    Returns:
        float: The effective rank or NaN if invalid.
    """
    if len(singular_values) == 0:
        return float('nan')
    s = np.array(singular_values)
    s = s / (s.sum() + 1e-10)
    entropy = -np.sum(s * np.log(s + 1e-10))
    return float(np.exp(entropy))

def compute_effective_rank_from_checkpoint(checkpoint_path: str) -> float:
    """
    Load a checkpoint and compute the effective rank of its embedding matrix.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint file.

    Returns:
        float: Effective rank or NaN if file missing.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state" in ckpt and "token_embed.weight" in ckpt["model_state"]:
            W = ckpt["model_state"]["token_embed.weight"].float().numpy()
            s = np.linalg.svd(W, compute_uv=False)
            return compute_effective_rank(s)
        return float('nan')
    except Exception:
        return float('nan')

def compute_weight_norm_trajectory(history: List[Dict]) -> List[float]:
    """
    Extract weight norm trajectory from history.

    Args:
        history (list of dict): Training history.

    Returns:
        list of float: The weight norms.
    """
    return [float(e.get("weight_norm", float('nan'))) for e in history]

def compute_train_test_gap_derivative(history: List[Dict], window: int = 5) -> Optional[float]:
    """
    Compute the derivative of the train-test loss gap using OLS over a recent window.

    Args:
        history (list of dict): Training history.
        window (int): Number of steps to compute slope over.

    Returns:
        float: The slope per 1000 steps, or None if history is too short.
    """
    if len(history) < window or len(history) < 2:
        return None

    recent = history[-window:]
    steps = np.array([e.get("step", 0) for e in recent], dtype=float)
    gaps = np.array([e.get("test_loss", 0) - e.get("train_loss", 0) for e in recent], dtype=float)

    if len(steps) < 2 or np.all(steps == steps[0]):
        return None

    A = np.column_stack([np.ones_like(steps), steps])
    sol, *_ = np.linalg.lstsq(A, gaps, rcond=None)
    return float(sol[1]) * 1000.0

def _get_attention_entropy(model, input_batch):
    """Internal helper to compute attention entropy by patching the transformer."""
    # To compute attention entropy without extraction hooks, we can just trace it
    # if we have the fastpath disabled, or simpler: since we just need entropy,
    # we can recreate the Q,K projection logic.
    # The ModularArithmeticTransformer has 1 layer.
    W_q, W_k, W_v = model.transformer.layers[0].self_attn.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = model.transformer.layers[0].self_attn.in_proj_bias.chunk(3, dim=0)

    # Forward pass up to attention
    tok = model.token_embed(input_batch)
    pos = model.pos_embed(torch.arange(2, device=input_batch.device).unsqueeze(0).expand(input_batch.shape[0], -1))
    h = tok + pos

    d_model = model.d_model
    n_heads = model.n_heads
    d_head = d_model // n_heads

    q = torch.nn.functional.linear(h, W_q, b_q)
    k = torch.nn.functional.linear(h, W_k, b_k)

    q = q.view(-1, 2, n_heads, d_head).transpose(1, 2) # (B, H, 2, D)
    k = k.view(-1, 2, n_heads, d_head).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5) # (B, H, 2, 2)
    attn = torch.nn.functional.softmax(scores, dim=-1)

    # Entropy over the key dimension (last dim)
    entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1) # (B, H, 2)

    # Return average entropy across heads, sequence, and batch
    return entropy.mean().item()

def compute_attention_entropy_slope(checkpoint_paths: List[str], config: Dict) -> Optional[float]:
    """
    Compute the slope of attention entropy across a sequence of checkpoints.

    Args:
        checkpoint_paths (list of str): Paths to checkpoints.
        config (dict): Model config to instantiate the model.

    Returns:
        float: Slope per 1000 steps, or None.
    """
    if len(checkpoint_paths) < 2:
        return None

    try:
        from src.model import ModularArithmeticTransformer
    except ImportError:
        return None

    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1)
    )
    model.eval()

    # Dummy input
    input_batch = torch.tensor([[0, 1], [10, 20], [30, 40], [58, 58]])

    steps = []
    entropies = []

    for path in checkpoint_paths:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            step = ckpt.get("step", 0)
            ent = _get_attention_entropy(model, input_batch)
            steps.append(float(step))
            entropies.append(float(ent))
        except Exception:
            continue

    if len(steps) < 2:
        return None

    steps_arr = np.array(steps)
    ent_arr = np.array(entropies)

    A = np.column_stack([np.ones_like(steps_arr), steps_arr])
    sol, *_ = np.linalg.lstsq(A, ent_arr, rcond=None)
    return float(sol[1]) * 1000.0
