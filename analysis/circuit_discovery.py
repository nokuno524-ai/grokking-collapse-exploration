import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Callable

from typing import Dict, List, Tuple, Callable, Union

def resample_ablation_hook(
    head_idx: int,
    clean_contributions: torch.Tensor,
    corrupt_contributions: torch.Tensor,
) -> Callable:
    """
    Creates a forward hook to patch the activation of a specific attention head
    with its corrupted version.
    """
    def hook(module: nn.Module, inputs: tuple, output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        is_tuple = isinstance(output, tuple)
        attn_out = output[0] if is_tuple else output

        # We subtract the clean head contribution and add the corrupt one
        diff = corrupt_contributions[:, :, head_idx, :] - clean_contributions[:, :, head_idx, :]
        attn_out = attn_out + diff

        if is_tuple:
            return (attn_out,) + output[1:]
        return attn_out

    return hook


def get_head_contributions(
    module: nn.MultiheadAttention,
    x: torch.Tensor,
    n_heads: int
) -> torch.Tensor:
    """
    Manually extracts the per-head contributions to the output of an nn.MultiheadAttention module.

    Args:
        module: The MultiheadAttention module.
        x: Input tensor. Expected shape is (batch_size, seq_len, d_model) if batch_first=True,
           else (seq_len, batch_size, d_model).
        n_heads: Number of attention heads.

    Returns:
        Tensor representing each head's contribution, reshaped back to match `x`.
        Output shape: (batch_size, seq_len, n_heads, d_model) or (seq_len, batch_size, n_heads, d_model).
    """
    batch_first = getattr(module, 'batch_first', False)
    if batch_first:
        batch_size, seq_len, d_model = x.shape
    else:
        seq_len, batch_size, d_model = x.shape
        x = x.transpose(0, 1) # Force batch first for calculation

    head_dim = d_model // n_heads

    # We assume self-attention (query = key = value = x)
    # Get projection weights
    qkv_weight = module.in_proj_weight
    qkv_bias = module.in_proj_bias

    out_weight = module.out_proj.weight

    if qkv_weight is None:
        raise ValueError("q_proj_weight, k_proj_weight, v_proj_weight not explicitly supported without in_proj_weight")

    q_w, k_w, v_w = qkv_weight.chunk(3, dim=0)
    q_b, k_b, v_b = qkv_bias.chunk(3, dim=0) if qkv_bias is not None else (None, None, None)

    # Project inputs
    q = (x @ q_w.T) + (q_b if q_b is not None else 0)
    k = (x @ k_w.T) + (k_b if k_b is not None else 0)
    v = (x @ v_w.T) + (v_b if v_b is not None else 0)

    # Reshape for multi-head attention: (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
    q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

    # Attention scores
    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)

    # Context vectors: (batch, n_heads, seq, head_dim)
    context = attn_weights @ v

    # Compute per-head contributions
    contributions = torch.zeros(batch_size, seq_len, n_heads, d_model, device=x.device)

    for h in range(n_heads):
        # The output projection weight is (d_model, d_model)
        # We take the columns corresponding to this head: (d_model, head_dim)
        W_O_h = out_weight[:, h * head_dim : (h + 1) * head_dim]

        # This head's context: (batch, seq, head_dim)
        head_context = context[:, h, :, :]

        # Project back to d_model: (batch, seq, head_dim) @ (head_dim, d_model) -> (batch, seq, d_model)
        contributions[:, :, h, :] = head_context @ W_O_h.T

    if not batch_first:
        contributions = contributions.transpose(0, 1) # Convert back to (seq_len, batch_size, n_heads, d_model)

    return contributions


def normalize_importance_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalizes importance scores to the range [0, 1] using min-max scaling.
    """
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

def compute_all_circuit_importances(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    corrupt_inputs: torch.Tensor,
    correct_labels: torch.Tensor,
    attention_layers: List[nn.MultiheadAttention],
    n_heads_per_layer: List[int]
) -> np.ndarray:
    """
    Computes importance scores for all heads across all provided layers.

    Returns:
        Normalized importance scores array of shape (n_layers, max_heads).
    """
    n_layers = len(attention_layers)
    max_heads = max(n_heads_per_layer)
    scores = np.zeros((n_layers, max_heads))

    for l_idx, layer in enumerate(attention_layers):
        n_heads = n_heads_per_layer[l_idx]
        for h_idx in range(n_heads):
            diff = compute_logit_diff(
                model, clean_inputs, corrupt_inputs, correct_labels,
                layer, n_heads, h_idx
            )
            scores[l_idx, h_idx] = diff

    return normalize_importance_scores(scores)


def compute_logit_diff(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    corrupt_inputs: torch.Tensor,
    correct_labels: torch.Tensor,
    layer_module: nn.MultiheadAttention,
    n_heads: int,
    head_idx: int
) -> float:
    """
    Computes the difference in logits when a specific attention head is patched.

    Args:
        model: The model.
        clean_inputs: Inputs for the clean run.
        corrupt_inputs: Inputs for the corrupt run (resample ablation).
        correct_labels: The correct class indices.
        layer_module: The specific nn.MultiheadAttention layer.
        n_heads: Total number of attention heads.
        head_idx: Index of the head to patch.

    Returns:
        The average logit difference.
    """
    model.eval()

    with torch.no_grad():
        # Baseline clean run
        clean_logits = model(clean_inputs)
        clean_correct_logits = clean_logits.gather(1, correct_labels.unsqueeze(-1)).squeeze(-1)

        # We need the inputs to `layer_module` for both clean and corrupt runs to calculate contributions
        clean_layer_inputs = []
        corrupt_layer_inputs = []

        def clean_pre_hook(module, args):
            clean_layer_inputs.append(args[0])

        def corrupt_pre_hook(module, args):
            corrupt_layer_inputs.append(args[0])

        h_clean = layer_module.register_forward_pre_hook(clean_pre_hook)
        _ = model(clean_inputs)
        h_clean.remove()

        h_corrupt = layer_module.register_forward_pre_hook(corrupt_pre_hook)
        _ = model(corrupt_inputs)
        h_corrupt.remove()

        clean_x = clean_layer_inputs[0]
        corrupt_x = corrupt_layer_inputs[0]

        clean_contributions = get_head_contributions(layer_module, clean_x, n_heads)
        corrupt_contributions = get_head_contributions(layer_module, corrupt_x, n_heads)

        # Now run patched model
        hook_fn = resample_ablation_hook(head_idx, clean_contributions, corrupt_contributions)
        h_patch = layer_module.register_forward_hook(hook_fn)

        patched_logits = model(clean_inputs)
        h_patch.remove()

        patched_correct_logits = patched_logits.gather(1, correct_labels.unsqueeze(-1)).squeeze(-1)

        # Logit difference (patched - clean)
        # If patching corrupts it, patched logits for correct label will be lower, so diff is negative.
        # Often "importance" is measured as clean - patched, so a larger positive value means important.
        logit_diff = (clean_correct_logits - patched_correct_logits).mean().item()

    return logit_diff

def generate_importance_heatmap(
    importance_scores: np.ndarray,
    save_path: str,
    title: str = "Attention Head Importance"
):
    """
    Generates a heatmap of head importance scores (e.g., layers x heads).

    Args:
        importance_scores: 2D array of shape (n_layers, n_heads) containing importance scores.
        save_path: Path to save the plot.
        title: Plot title.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(importance_scores, cmap='viridis', aspect='auto')
    plt.colorbar(label='Importance (Logit Diff)')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.title(title)

    # Add text annotations
    for i in range(importance_scores.shape[0]):
        for j in range(importance_scores.shape[1]):
            plt.text(j, i, f"{importance_scores[i, j]:.2f}",
                     ha="center", va="center", color="w" if importance_scores[i, j] < importance_scores.max()/2 else "k")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
