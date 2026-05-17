"""
Mechanistic Interpretability Dashboard.
Visualizes logit attribution, attention specialization, and circuit formation.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_logit_attribution(history: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Plot component-level logit attribution over training.
    Requires history items to have a 'logit_attribution' key from the tracking loop.

    Args:
        history: List of dictionaries containing metrics over time
        output_path: Path to save the figure
    """
    if not history:
        return

    steps = [h.get('step', i) for i, h in enumerate(history)]

    # Extract attributions if they exist in history, otherwise use a fallback mock for demonstration
    has_attr = any('logit_attribution' in h for h in history)

    if has_attr:
        tok_attr = [h.get('logit_attribution', {}).get('token_embed_direct_norm', 0.0) for h in history]
        pos_attr = [h.get('logit_attribution', {}).get('pos_embed_direct_norm', 0.0) for h in history]
    else:
        # Fallback if the training loop wasn't updated to save attributions yet
        # Uses weight norm and embedding rank as structural proxies
        tok_attr = [h.get('weight_norm', 0.0) for h in history]
        pos_attr = [h.get('embedding_rank', 0.0) for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Token Embed Direct Path Norm' if has_attr else 'Weight Norm', color=color1)
    ax1.plot(steps, tok_attr, color=color1, label='Token Embed Path' if has_attr else 'Weight Norm')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Pos Embed Direct Path Norm' if has_attr else 'Embedding Rank', color=color2)
    ax2.plot(steps, pos_attr, color=color2, label='Pos Embed Path' if has_attr else 'Embedding Rank')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Logit Attribution Over Training' if has_attr else 'Structural Proxies Over Training')
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_head_specialization(history: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Plot attention head specialization tracking over time.
    Requires 'head_metrics' in history dicts.

    Args:
        history: List of dictionaries containing metrics over time
        output_path: Path to save the figure
    """
    if not history:
        return

    steps = [h.get('step', i) for i, h in enumerate(history)]

    has_heads = any('head_metrics' in h for h in history)

    plt.figure(figsize=(10, 6))

    if has_heads:
        # Assuming head_metrics is a dict mapping head index to some specialization score
        first_heads = next((h['head_metrics'] for h in history if 'head_metrics' in h), {})
        for head_idx in first_heads.keys():
            head_scores = [h.get('head_metrics', {}).get(head_idx, 0.0) for h in history]
            plt.plot(steps, head_scores, label=f'Head {head_idx}')
        plt.ylabel('Head Specialization Score')
        plt.legend()
    else:
        # Fallback to Fourier concentration
        fourier_conc = [h.get('fourier_concentration', 0.0) for h in history]
        plt.plot(steps, fourier_conc, 'g-', linewidth=2)
        plt.ylabel('Fourier Concentration')

    plt.xlabel('Training Step')
    plt.title('Attention Head Specialization' if has_heads else 'Representation Specialization')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
def plot_circuit_formation(
    grokked_history: List[Dict[str, Any]],
    collapsed_history: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Plot circuit formation timeline comparing grokking vs collapse.

    Args:
        grokked_history: History from a grokked model
        collapsed_history: History from a collapsed model
        output_path: Path to save the figure
    """
    if not grokked_history or not collapsed_history:
        return

    g_steps = [h.get('step', i) for i, h in enumerate(grokked_history)]
    c_steps = [h.get('step', i) for i, h in enumerate(collapsed_history)]

    g_acc = [h.get('test_acc', 0.0) for h in grokked_history]
    c_acc = [h.get('test_acc', 0.0) for h in collapsed_history]

    g_fourier = [h.get('fourier_concentration', 0.0) for h in grokked_history]
    c_fourier = [h.get('fourier_concentration', 0.0) for h in collapsed_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Accuracy comparison
    ax1.plot(g_steps, g_acc, 'b-', label='Grokked - Test Acc')
    ax1.plot(c_steps, c_acc, 'r--', label='Collapsed - Test Acc')
    ax1.axhline(0.95, color='k', linestyle=':', label='Grokking Threshold')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Task Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fourier concentration (Circuit formation proxy) comparison
    ax2.plot(g_steps, g_fourier, 'b-', label='Grokked - Fourier Conc')
    ax2.plot(c_steps, c_fourier, 'r--', label='Collapsed - Fourier Conc')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Fourier Concentration')
    ax2.set_title('Circuit Formation (Fourier Basis)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Circuit Formation: Grokking vs Collapse', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
