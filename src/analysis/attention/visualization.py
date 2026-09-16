import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def plot_attention_heatmaps(attn_matrix: np.ndarray, out_path: Path):
    """
    Plots attention heatmaps for each head in a single batch sample.
    attn_matrix shape: (batch_size, n_heads, seq_len, seq_len)
    """
    n_heads = attn_matrix.shape[1]
    sample = attn_matrix[0] # Take first sample

    fig, axes = plt.subplots(1, n_heads, figsize=(3*n_heads, 3))
    if n_heads == 1:
        axes = [axes]

    for i in range(n_heads):
        im = axes[i].imshow(sample[i], cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(f'Head {i}')
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xlabel('Key')
        axes[i].set_ylabel('Query')

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
