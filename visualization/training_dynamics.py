import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional

def detect_grokking_transition(accuracies: List[float], random_threshold: float = 0.1, perfect_threshold: float = 0.95) -> Optional[Tuple[int, int]]:
    """
    Detect the step range where accuracy jumps from near-random to near-perfect.

    Args:
        accuracies: List of accuracy values across training steps
        random_threshold: Accuracy below which is considered random performance
        perfect_threshold: Accuracy above which is considered grokked

    Returns:
        Tuple of (start_idx, end_idx) where transition happens, or None if not found
    """
    if not accuracies:
        return None

    start_idx = None
    end_idx = None

    # Find grokking point (first time > perfect_threshold)
    for i, acc in enumerate(accuracies):
        if acc > perfect_threshold:
            end_idx = i
            break

    if end_idx is None:
        return None  # Never grokked

    # Walk backward to find where it was still "random" or flat
    for i in range(end_idx, -1, -1):
        if accuracies[i] < random_threshold:
            start_idx = i
            break

    if start_idx is None:
        start_idx = 0

    return (start_idx, end_idx)

def plot_training_dynamics(history: List[Dict], save_path: str):
    """
    Create a multi-panel figure for loss, accuracy, weight norm, etc. over time.

    Args:
        history: List of dicts, each with keys like 'step', 'train_loss', 'test_loss',
                 'train_acc', 'test_acc', 'weight_norm'
        save_path: Where to save the plot
    """
    steps = [h['step'] for h in history]
    train_loss = [h.get('train_loss', float('nan')) for h in history]
    test_loss = [h.get('test_loss', float('nan')) for h in history]
    train_acc = [h.get('train_acc', float('nan')) for h in history]
    test_acc = [h.get('test_acc', float('nan')) for h in history]
    weight_norm = [h.get('weight_norm', float('nan')) for h in history]

    # Optional metrics
    l0_sparsity = [h.get('l0_sparsity', float('nan')) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Loss
    axes[0].plot(steps, train_loss, label='Train Loss', alpha=0.7)
    axes[0].plot(steps, test_loss, label='Test Loss', alpha=0.7)
    axes[0].set_yscale('log')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].legend()

    # Accuracy
    axes[1].plot(steps, train_acc, label='Train Acc', alpha=0.7)
    axes[1].plot(steps, test_acc, label='Test Acc', alpha=0.7)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # Detect transition and highlight it
    transition = detect_grokking_transition(test_acc)
    if transition:
        start_step = steps[transition[0]]
        end_step = steps[transition[1]]
        axes[1].axvspan(start_step, end_step, color='gray', alpha=0.2, label='Transition')
        axes[1].legend()

    # Weight Norm
    axes[2].plot(steps, weight_norm, color='purple')
    axes[2].set_title('Weight Norm')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('L2 Norm')

    # L0 Sparsity or placeholder
    if not np.isnan(l0_sparsity).all():
        axes[3].plot(steps, l0_sparsity, color='green')
        axes[3].set_title('L0 Sparsity')
        axes[3].set_ylabel('Fraction Non-Zero')
    else:
        # Fallback to fourier or rank if L0 isn't available
        fourier = [h.get('fourier_concentration', float('nan')) for h in history]
        if not np.isnan(fourier).all():
            axes[3].plot(steps, fourier, color='orange')
            axes[3].set_title('Fourier Concentration')
            axes[3].set_ylabel('Concentration')
        else:
            axes[3].text(0.5, 0.5, 'No sparsity/fourier data', ha='center', va='center')
    axes[3].set_xlabel('Step')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_fourier_evolution(histories: Dict[str, List[Dict]], save_path: str):
    """
    Show Fourier concentration evolution over training.

    Args:
        histories: Dict mapping condition names to their history lists
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))

    for condition, history in histories.items():
        steps = [h['step'] for h in history]
        fourier = [h.get('fourier_concentration', float('nan')) for h in history]
        plt.plot(steps, fourier, label=condition)

    plt.title('Fourier Concentration over Training')
    plt.xlabel('Step')
    plt.ylabel('Top-k Fourier Energy Concentration')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_condition_overlay(results_dict: Dict[str, List[Dict]], metric: str, save_path: str):
    """
    Overlay a specific metric (e.g. test_acc) from different conditions.

    Args:
        results_dict: Dict mapping condition names to history lists
        metric: Which metric to plot (e.g. 'test_acc', 'weight_norm')
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))

    for condition, history in results_dict.items():
        steps = [h['step'] for h in history]
        vals = [h.get(metric, float('nan')) for h in history]
        plt.plot(steps, vals, label=condition)

    plt.title(f'{metric} over Training Across Conditions')
    plt.xlabel('Step')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
