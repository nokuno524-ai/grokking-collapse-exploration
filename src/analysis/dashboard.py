import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional

def generate_comparison_dashboard(
    loss_data: Dict[str, Dict[str, np.ndarray]],
    weight_norm_data: Dict[str, Dict[str, np.ndarray]],
    attention_entropy_data: Dict[str, Dict[str, np.ndarray]],
    grokking_prob_matrix: np.ndarray,
    collapse_levels: List[str],
    model_sizes: List[str],
    fourier_spectra: Dict[str, np.ndarray],
    save_path: str = "results/dashboard/comparison_dashboard.pdf"
):
    """
    Generate a publication-quality multi-panel figure comparing:
    (a) training loss curves with grokking annotations
    (b) weight norm trajectories across collapse levels
    (c) attention pattern entropy evolution
    (d) grokking probability heatmap across collapse_level x model_size
    (e) Fourier power spectrum of embeddings
    """
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # Set style for publication quality
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.2)

    fig = plt.figure(figsize=(15, 10))

    # Create 2x3 grid
    gs = fig.add_gridspec(2, 3)

    colors = sns.color_palette("viridis", len(collapse_levels))
    color_map = dict(zip(collapse_levels, colors))

    # (a) Training loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    for level in collapse_levels:
        if level in loss_data:
            steps = loss_data[level]['steps']
            loss = loss_data[level]['loss']
            ax1.plot(steps, loss, label=level, color=color_map.get(level, 'black'))

            # Annotate grokking point if provided
            if 'grokking_step' in loss_data[level] and not np.isnan(loss_data[level]['grokking_step']):
                grok_step = loss_data[level]['grokking_step']
                # Find index closest to grok_step
                idx = np.abs(np.array(steps) - grok_step).argmin()
                ax1.plot(grok_step, loss[idx], 'r*', markersize=10)

    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Test Loss')
    ax1.set_yscale('log')
    ax1.set_title('(a) Loss Curves & Grokking')
    ax1.legend(fontsize=8)

    # (b) Weight norm trajectories
    ax2 = fig.add_subplot(gs[0, 1])
    for level in collapse_levels:
        if level in weight_norm_data:
            steps = weight_norm_data[level]['steps']
            norms = weight_norm_data[level]['norms']
            ax2.plot(steps, norms, label=level, color=color_map.get(level, 'black'))

    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('L2 Weight Norm')
    ax2.set_title('(b) Weight Norm Evolution')

    # (c) Attention pattern entropy
    ax3 = fig.add_subplot(gs[0, 2])
    for level in collapse_levels:
        if level in attention_entropy_data:
            steps = attention_entropy_data[level]['steps']
            entropy = attention_entropy_data[level]['entropy']
            ax3.plot(steps, entropy, label=level, color=color_map.get(level, 'black'))

    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Attention Entropy')
    ax3.set_title('(c) Attention Entropy')

    # (d) Grokking probability heatmap
    ax4 = fig.add_subplot(gs[1, 0:2])
    sns.heatmap(
        grokking_prob_matrix,
        annot=True,
        cmap='YlGnBu',
        xticklabels=model_sizes,
        yticklabels=collapse_levels,
        ax=ax4,
        vmin=0, vmax=1
    )
    ax4.set_xlabel('Model Size')
    ax4.set_ylabel('Collapse Level')
    ax4.set_title('(d) Grokking Probability P(Grok | Collapse, Size)')

    # (e) Fourier power spectrum
    ax5 = fig.add_subplot(gs[1, 2])
    for level in collapse_levels:
        if level in fourier_spectra:
            spectrum = fourier_spectra[level]
            freqs = np.arange(len(spectrum))
            ax5.plot(freqs, spectrum, label=level, color=color_map.get(level, 'black'))

    ax5.set_xlabel('Frequency Component')
    ax5.set_ylabel('Power')
    ax5.set_title('(e) Embedding Fourier Spectrum')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path
