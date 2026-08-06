import numpy as np
from typing import List, Tuple, Dict, Optional

def detect_grokking_point(train_loss: List[float], val_loss: List[float], threshold_ratio: float = 2.0) -> int:
    """
    Detects the grokking transition point where validation loss drops sharply after training loss
    has already converged, or simply where validation loss decoupling ends.

    Args:
        train_loss: List of training loss values per step.
        val_loss: List of validation loss values per step.
        threshold_ratio: Ratio to consider a significant drop.

    Returns:
        The index (step) where grokking occurs, or -1 if not detected.
    """
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    if len(train_loss) < 2:
        return -1

    # We look for a point where validation loss experiences a sharp drop
    # Often found by looking at the discrete derivative of val loss.
    val_loss_diff = np.diff(val_loss)

    # Simple heuristic: find max negative derivative of val_loss
    # but only if val_loss drops below a certain threshold overall.
    if np.min(val_loss) > 0.5 * np.max(val_loss):
        return -1 # no significant grokking

    # Smoothen the difference slightly if needed, but we'll use argmin directly
    grokking_idx = np.argmin(val_loss_diff) + 1

    return grokking_idx

def detect_collapse_onset(weight_norms: List[float], drop_threshold: float = 0.1) -> int:
    """
    Detects the onset of collapse by finding significant drops in weight norm.

    Args:
        weight_norms: List of weight norms per step.
        drop_threshold: Threshold for relative drop in weight norm.

    Returns:
        The index where collapse onset occurs, or -1 if not detected.
    """
    if len(weight_norms) < 2:
        return -1

    wn = np.array(weight_norms)
    # A relative drop of > 10% (by default) might indicate collapse onset
    relative_drops = (wn[:-1] - wn[1:]) / (wn[:-1] + 1e-9)

    onset_idx = np.where(relative_drops > drop_threshold)[0]
    if len(onset_idx) > 0:
        return int(onset_idx[0] + 1)

    return -1

def grokking_gap_metrics(train_acc: List[float], val_acc: List[float], memo_threshold: float = 0.95, gen_threshold: float = 0.95) -> int:
    """
    Computes the "grokking gap" (steps between memorization and generalization).

    Args:
        train_acc: List of training accuracies.
        val_acc: List of validation accuracies.
        memo_threshold: Threshold to consider memorization complete.
        gen_threshold: Threshold to consider generalization complete.

    Returns:
        The number of steps between memorization and generalization, or -1 if generalization doesn't occur.
    """
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)

    memo_idx = np.where(train_acc >= memo_threshold)[0]
    gen_idx = np.where(val_acc >= gen_threshold)[0]

    if len(memo_idx) == 0:
        return -1
    if len(gen_idx) == 0:
        return -1

    first_memo = memo_idx[0]
    first_gen = gen_idx[0]

    gap = first_gen - first_memo
    return gap if gap > 0 else 0
