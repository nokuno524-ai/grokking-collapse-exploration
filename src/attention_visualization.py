import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import math

def plot_comparison_heatmaps(pure_weights: torch.Tensor,
                             collapsed_weights: torch.Tensor,
                             step: int,
                             output_path: Path | str,
                             layer_idx: int = 0) -> None:
    """
    Plot side-by-side heatmaps of pure vs collapsed attention weights.
    """
    pure_mean = pure_weights.mean(dim=0).cpu().numpy()
    col_mean = collapsed_weights.mean(dim=0).cpu().numpy()

    n_heads = pure_mean.shape[0]
    seq_len = pure_mean.shape[1]

    fig, axes = plt.subplots(2, n_heads, figsize=(3 * n_heads, 6))

    if n_heads == 1:
        axes = np.array([axes]).T

    for h in range(n_heads):
        # Pure
        ax = axes[0, h]
        im = ax.imshow(pure_mean[h], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Pure Head {h}')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))

        # Collapsed
        ax = axes[1, h]
        im = ax.imshow(col_mean[h], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Collapsed Head {h}')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))

    fig.suptitle(f'Attention Heatmaps - Layer {layer_idx}, Step {step}')
    plt.tight_layout()

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.subplots_adjust(right=0.9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_evolution_frames(weights_history: Dict[int, torch.Tensor],
                          output_dir: Path | str,
                          prefix: str = "attn",
                          layer_idx: int = 0) -> None:
    """
    Generate frames showing attention evolution over steps.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step, weights in sorted(weights_history.items()):
        w_mean = weights.mean(dim=0).cpu().numpy()
        n_heads = w_mean.shape[0]
        seq_len = w_mean.shape[1]

        fig, axes = plt.subplots(1, n_heads, figsize=(3 * n_heads, 3.5))
        if n_heads == 1:
            axes = [axes]

        for h in range(n_heads):
            ax = axes[h]
            im = ax.imshow(w_mean[h], cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'Head {h}')
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))

        fig.suptitle(f'Step {step}')
        plt.tight_layout()

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.subplots_adjust(right=0.9)

        plt.savefig(output_dir / f"{prefix}_step_{step:06d}.png", dpi=150)
        plt.close(fig)

def plot_metrics_over_time(metrics_history: Dict[str, Dict[int, torch.Tensor]],
                           output_path_prefix: Path | str,
                           labels: List[str]) -> None:
    """
    Plot scalar attention metrics over training time and save to CSV.
    """
    out_base = Path(output_path_prefix)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ['entropy', 'diag_ratio', 'specialization']
    csv_data = []

    for metric_name in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cond_label in labels:
            if cond_label not in metrics_history:
                continue

            steps = sorted(metrics_history[cond_label][metric_name].keys())
            if not steps:
                continue

            means = []
            stds = []

            for step in steps:
                val = metrics_history[cond_label][metric_name][step].float().cpu()
                val_l0 = val[0]

                mean_val = val_l0.mean().item()
                std_val = val_l0.std().item() if val_l0.numel() > 1 else 0.0

                means.append(mean_val)
                stds.append(std_val)

                csv_data.append({
                    'condition': cond_label,
                    'metric': metric_name,
                    'layer': 0,
                    'step': step,
                    'mean': mean_val,
                    'std': std_val
                })

            ax.plot(steps, means, label=cond_label)
            ax.fill_between(steps,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.2)

        ax.set_title(f'{metric_name.capitalize()} Over Training (Layer 0)')
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{out_base}_{metric_name}.png", dpi=150)
        plt.close(fig)

    if csv_data:
        csv_path = f"{out_base}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['condition', 'metric', 'layer', 'step', 'mean', 'std'])
            writer.writeheader()
            writer.writerows(csv_data)