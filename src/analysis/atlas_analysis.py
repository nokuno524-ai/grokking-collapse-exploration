import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import io
import base64

def render_atlas_figure(attn_weights: np.ndarray, title: str = "Attention Atlas") -> str:
    """
    Renders a layer x head "atlas" figure of attention weights as a base64 encoded PNG.

    Args:
        attn_weights: numpy array of shape (num_heads, tgt_len, src_len)
                      (Currently handles single layer for this task).
        title: Title of the figure.

    Returns:
        Base64 string of the generated PNG image.
    """
    num_heads = attn_weights.shape[0]

    fig, axes = plt.subplots(1, num_heads, figsize=(3 * num_heads, 3))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn_weights[h], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Head {h}")
        ax.set_xlabel("Source Token")
        ax.set_ylabel("Target Token")

    plt.suptitle(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64
