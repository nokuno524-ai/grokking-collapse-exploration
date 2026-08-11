import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from pathlib import Path
from typing import Dict

def plot_feature_activations(activations: Dict[str, torch.Tensor], output_dir: Path):
    """Plot distribution of feature activations across layers/heads."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for name, acts in activations.items():
        flat_acts = acts.detach().cpu().numpy().flatten()
        sns.kdeplot(flat_acts, label=name, fill=True, alpha=0.4)

    plt.title("Feature Activation Distribution Comparisons")
    plt.xlabel("Activation Value")
    plt.ylabel("Density")
    plt.legend()

    out = output_dir / "feature_activations"
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches='tight')
    plt.savefig(out.with_suffix(".pdf"), dpi=150, bbox_inches='tight')
    plt.close()
