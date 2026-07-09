import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from matplotlib import cm

def plot_3d_accuracy_surface(
    collapse_levels: List[float],
    steps: List[int],
    accuracies: np.ndarray,
    save_path: str
):
    """
    3D surface plot showing accuracy as function of collapse level and training step.

    Args:
        collapse_levels: 1D array/list of collapse levels (Y axis)
        steps: 1D array/list of training steps (X axis)
        accuracies: 2D array of accuracies of shape (len(collapse_levels), len(steps))
        save_path: Output file path
    """
    X, Y = np.meshgrid(steps, collapse_levels)
    Z = accuracies

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                           linewidth=0, antialiased=True)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Collapse Level')
    ax.set_zlabel('Test Accuracy')
    ax.set_title('Test Accuracy Surface vs. Collapse Level')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_grokking_onset_heatmap(
    collapse_levels: List[float],
    severities: List[float],
    onset_steps: np.ndarray,
    save_path: str
):
    """
    Heatmap of grokking onset step vs collapse severity and level.

    Args:
        collapse_levels: Y-axis values
        severities: X-axis values
        onset_steps: 2D array of grokking onset steps
        save_path: Output path
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Mask out None/NaN values (didn't grok)
    masked_data = np.ma.masked_invalid(onset_steps)
    cmap = cm.viridis.copy()
    cmap.set_bad(color='black')  # Black for no grokking

    im = ax.imshow(masked_data, cmap=cmap, aspect='auto', origin='lower')

    ax.set_xticks(np.arange(len(severities)))
    ax.set_yticks(np.arange(len(collapse_levels)))
    ax.set_xticklabels([f"{s:.2f}" for s in severities])
    ax.set_yticklabels([f"{c:.2f}" for c in collapse_levels])

    ax.set_xlabel('Collapse Severity')
    ax.set_ylabel('Collapse Level')
    ax.set_title('Grokking Onset Step (Black = No Grokking)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Training Step')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_weight_norm_trajectory(
    histories: Dict[str, List[Dict]],
    save_path: str,
    confidence_intervals: Dict[str, Tuple[List[float], List[float]]] = None
):
    """
    Plot weight norm trajectory comparison across collapse levels,
    optionally with statistical significance markers (shaded CIs).

    Args:
        histories: Dict mapping collapse condition name to history list
        save_path: Output path
        confidence_intervals: Dict mapping condition name to (lower_bound_list, upper_bound_list)
    """
    plt.figure(figsize=(8, 6))

    for condition, history in histories.items():
        steps = [h['step'] for h in history]
        norms = [h.get('weight_norm', float('nan')) for h in history]

        line, = plt.plot(steps, norms, label=condition)

        # Add shaded confidence intervals if provided
        if confidence_intervals and condition in confidence_intervals:
            lower, upper = confidence_intervals[condition]
            plt.fill_between(steps, lower, upper, color=line.get_color(), alpha=0.2)

    plt.title('Weight Norm Trajectory Across Conditions')
    plt.xlabel('Step')
    plt.ylabel('Weight Norm (L2)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
