#!/usr/bin/env python3
"""
Attention visualization suite.
Loads a checkpoint (or sweep of checkpoints), extracts attention matrices,
and generates plots: heatmaps, entropy trends, clustering, Fourier probes,
and head ablation ranking.
"""
import os
import argparse
import glob
import re
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import umap
from typing import List, Dict, Any, Tuple, Optional
import copy

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ModularArithmeticTransformer
from src.data import generate_modular_arithmetic, DatasetConfig
from src.attention_extraction import extract_attention

def load_checkpoint(ckpt_path: str, device: str = 'cpu') -> Tuple[ModularArithmeticTransformer, Dict[str, Any]]:
    """Loads a ModularArithmeticTransformer from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get('config', {})

    # Handle older checkpoints missing these
    prime = config.get('prime', 59)
    d_model = config.get('d_model', 128)
    n_heads = config.get('n_heads', 4)
    d_ff = config.get('d_ff', 512)
    n_layers = config.get('n_layers', 1)

    model = ModularArithmeticTransformer(
        prime=prime,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
    )

    state_dict = ckpt.get('model_state_dict', ckpt)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, config

def chunked_eval(model: ModularArithmeticTransformer, x: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Evaluate model in chunks to avoid OOM."""
    outs = []
    for i in range(0, x.shape[0], chunk_size):
        chunk_x = x[i:i+chunk_size]
        with torch.no_grad():
            outs.append(model(chunk_x))
    return torch.cat(outs, dim=0)

def extract_attention_patterns(model: ModularArithmeticTransformer, x: torch.Tensor, chunk_size: int = 1024) -> np.ndarray:
    """
    Extracts attention patterns. Processes in chunks to prevent OOM.
    Returns array of shape (n_layers, batch_size, n_heads, seq_len, seq_len).
    """
    device = next(model.parameters()).device
    all_attns = []
    for i in range(0, x.shape[0], chunk_size):
        chunk_x = x[i:i+chunk_size].to(device)
        with torch.no_grad():
            with extract_attention() as attentions:
                _ = model(chunk_x)
        # attentions is a list of layers, each (chunk_size, n_heads, seq, seq)
        # We append to all_attns
        all_attns.append(attentions)

    # Reassemble across chunks
    # all_attns[i] is a list of layers
    if not all_attns:
        return np.array([])

    n_layers = len(all_attns[0])
    combined_layers = []
    for l in range(n_layers):
        layer_chunks = [chunk[l] for chunk in all_attns]
        combined_layers.append(np.concatenate(layer_chunks, axis=0))

    return np.array(combined_layers)  # (n_layers, batch, n_heads, seq, seq)

def compute_entropy(attn_weights: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Computes entropy of attention distributions over the key sequence.
    attn_weights: (..., seq_len, seq_len)
    Returns entropy: (...)
    """
    # Sum over the last dimension (keys)
    return -np.sum(attn_weights * np.log(np.clip(attn_weights, eps, 1.0)), axis=-1).mean(axis=-1)

def plot_attention_heatmaps(attentions: np.ndarray, save_path: str):
    """
    Plots attention heatmaps per layer and head, averaged over the batch.
    attentions: (n_layers, batch, n_heads, seq, seq)
    """
    if attentions.size == 0:
        print("Empty attention array, skipping heatmap.")
        return

    n_layers, _, n_heads, seq_len, _ = attentions.shape
    avg_attn = attentions.mean(axis=1) # (n_layers, n_heads, seq, seq)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 3, n_layers * 3), squeeze=False)

    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            im = ax.imshow(avg_attn[l, h], cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'Layer {l} Head {h}')
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_entropy_over_training(ckpt_paths: List[str], steps: List[int], x: torch.Tensor, save_path: str, device: str = 'cpu'):
    """
    Computes and plots attention entropy per head across training steps.
    """
    all_entropies = [] # list of (n_layers, n_heads)

    for ckpt_path in ckpt_paths:
        model, _ = load_checkpoint(ckpt_path, device=device)
        attns = extract_attention_patterns(model, x) # (L, B, H, S, S)
        # Compute entropy per batch item, then mean over batch
        # entropy -> (L, B, H)
        entropy = compute_entropy(attns).mean(axis=1) # (L, H)
        all_entropies.append(entropy)

    all_entropies = np.array(all_entropies) # (n_steps, L, H)
    n_steps, n_layers, n_heads = all_entropies.shape

    fig, axes = plt.subplots(n_layers, 1, figsize=(8, 4 * n_layers), squeeze=False)

    for l in range(n_layers):
        ax = axes[l, 0]
        for h in range(n_heads):
            ax.plot(steps, all_entropies[:, l, h], label=f'Head {h}', marker='o', markersize=3)
        ax.set_title(f'Layer {l} Attention Entropy')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Entropy (nats)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_head_clustering(ckpt_paths: List[str], collapse_levels: List[float], steps: List[int], x: torch.Tensor, save_path: str, device: str = 'cpu'):
    """
    Uses UMAP to cluster flattened attention patterns across different checkpoints.
    Colors by collapse severity (e.g., noise level).
    """
    patterns = []
    labels = []

    for ckpt_path, severity, step in zip(ckpt_paths, collapse_levels, steps):
        model, _ = load_checkpoint(ckpt_path, device=device)
        attns = extract_attention_patterns(model, x) # (L, B, H, S, S)

        # We average over batch to get (L, H, S, S), then flatten each head
        avg_attns = attns.mean(axis=1) # (L, H, S, S)
        n_layers, n_heads, seq_len, _ = avg_attns.shape

        for l in range(n_layers):
            for h in range(n_heads):
                flat_pattern = avg_attns[l, h].flatten()
                patterns.append(flat_pattern)
                labels.append(severity)

    if not patterns:
        return

    patterns = np.array(patterns)

    # Needs enough samples for UMAP, fallback to PCA or skip if too small
    if len(patterns) < 5:
        print("Not enough patterns for UMAP clustering.")
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        n_neighbors = min(15, len(patterns) - 1)
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
        embedding = reducer.fit_transform(patterns)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7, edgecolors='none')
    ax.set_title('UMAP Clustering of Head Attention Patterns')
    plt.colorbar(sc, ax=ax, label='Collapse Severity (Noise)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def fourier_probe(activations: np.ndarray) -> float:
    """
    Computes 2D Fourier transform of attention pattern to detect periodic structure.
    activations: (seq_len, seq_len)
    Returns peak-significance score (max mag / mean mag, excluding DC).
    """
    fft2d = np.fft.fft2(activations)
    mag = np.abs(fft2d)
    # Zero out DC component
    mag[0, 0] = 0
    mean_val = mag.mean()
    if mean_val <= 1e-10:
        return 0.0
    return mag.max() / mean_val

def plot_fourier_scores(attentions: np.ndarray, save_path: str):
    """
    Computes and plots Fourier significance scores for each layer/head.
    attentions: (n_layers, batch, n_heads, seq_len, seq_len)
    """
    if attentions.size == 0:
        return

    n_layers, _, n_heads, seq_len, _ = attentions.shape
    scores = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        for h in range(n_heads):
            # Average attention over batch before FFT
            head_attn = attentions[l, :, h].mean(axis=0)
            scores[l, h] = fourier_probe(head_attn)

    fig, ax = plt.subplots(figsize=(6, max(4, n_layers)))
    im = ax.imshow(scores, cmap='magma')
    ax.set_title('Fourier Peak Significance per Head')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))

    # Add text annotations
    for l in range(n_layers):
        for h in range(n_heads):
            ax.text(h, l, f"{scores[l, h]:.1f}", ha="center", va="center",
                    color="w" if scores[l, h] < (scores.max() + scores.min()) / 2 else "k")

    fig.colorbar(im, ax=ax, label='Significance Score')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def ablate_head_and_evaluate(
    model: ModularArithmeticTransformer,
    layer_idx: int,
    head_idx: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """
    Evaluates the model on train and test loaders with a specific head ablated.
    Head ablation zeroes out the output of the head.
    """
    # Get the transformer layer
    layer = model.transformer.layers[layer_idx]
    mha = layer.self_attn

    def out_proj_pre_hook(module, args):
        # args is a tuple, args[0] is the concatenated head outputs (batch, seq, d_model)
        x = args[0].clone()
        d_head = model.d_model // model.n_heads
        start = head_idx * d_head
        end = start + d_head
        x[..., start:end] = 0
        return (x,)

    # Register hook
    hook_handle = mha.out_proj.register_forward_pre_hook(out_proj_pre_hook)

    model.eval()

    def evaluate(loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)

    # Remove hook
    hook_handle.remove()

    return train_acc, test_acc

def rank_heads_by_ablation(
    model: ModularArithmeticTransformer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    save_path: str,
    device: str = 'cpu'
):
    """
    Ranks heads by their importance to generalization using Leave-One-Head-Out (LOHO) ablation.
    Outputs a ranked Markdown table.
    """
    # Get base accuracy
    def evaluate(loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    base_train = evaluate(train_loader)
    base_test = evaluate(test_loader)

    results = []

    for l in range(len(model.transformer.layers)):
        for h in range(model.n_heads):
            train_acc, test_acc = ablate_head_and_evaluate(model, l, h, train_loader, test_loader, device)
            train_drop = base_train - train_acc
            test_drop = base_test - test_acc

            results.append({
                'layer': l,
                'head': h,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_drop': train_drop,
                'test_drop': test_drop
            })

    # Sort by test_drop descending (most important for test acc first)
    results.sort(key=lambda x: x['test_drop'], reverse=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("# Head Ablation Ranking (LOHO)\n\n")
        f.write(f"**Base Train Acc**: {base_train:.4f} | **Base Test Acc**: {base_test:.4f}\n\n")
        f.write("| Rank | Layer | Head | Train Acc | Test Acc | Train Drop | Test Drop |\n")
        f.write("|------|-------|------|-----------|----------|------------|-----------|\n")
        for i, res in enumerate(results):
            f.write(f"| {i+1} | {res['layer']} | {res['head']} | {res['train_acc']:.4f} | {res['test_acc']:.4f} | {res['train_drop']:.4f} | {res['test_drop']:.4f} |\n")

if __name__ == "__main__":
    pass
