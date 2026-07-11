import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

def compute_head_importance(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device = None) -> torch.Tensor:
    """
    Compute importance score for each attention head via causal patching (zero-ablation).

    Args:
        model: ModularArithmeticTransformer model
        dataloader: DataLoader for evaluation data
        device: Device to use for computation

    Returns:
        Tensor of shape (n_layers, n_heads) containing importance scores (drop in accuracy)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Check if model has the expected structure
    if not hasattr(model, 'transformer') or not hasattr(model, 'n_heads'):
        raise ValueError("Model does not have expected 'transformer' or 'n_heads' attributes")

    n_layers = len(model.transformer.layers)
    n_heads = model.n_heads
    d_model = model.d_model
    d_head = d_model // n_heads

    importance = torch.zeros((n_layers, n_heads), device=device)

    # 1. Compute baseline accuracy
    correct_base = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            correct_base += (preds == y).sum().item()
            total += y.size(0)

    base_acc = correct_base / max(1, total)

    # 2. Iterate through each layer and head, ablate, and measure accuracy drop
    with torch.no_grad():
        for layer_idx in range(n_layers):
            layer = model.transformer.layers[layer_idx]
            attn = layer.self_attn

            # Save original out_proj weight
            orig_weight = attn.out_proj.weight.clone()

            for head_idx in range(n_heads):
                # Zero out the contribution of this head by setting its corresponding columns in out_proj to 0
                attn.out_proj.weight.data[:, head_idx*d_head : (head_idx+1)*d_head] = 0

                correct_ablated = 0
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    preds = torch.argmax(logits, dim=-1)
                    correct_ablated += (preds == y).sum().item()

                ablated_acc = correct_ablated / max(1, total)

                # Importance is drop in accuracy (larger drop = more important)
                importance[layer_idx, head_idx] = base_acc - ablated_acc

                # Restore original weight
                attn.out_proj.weight.data.copy_(orig_weight)

    return importance

def track_head_importance_over_training(checkpoint_paths: List[str], model: nn.Module,
                                        dataloader: torch.utils.data.DataLoader,
                                        device: torch.device = None) -> Tuple[List[int], torch.Tensor]:
    """
    Track attention head importance across multiple training checkpoints.

    Args:
        checkpoint_paths: List of paths to checkpoint .pt files
        model: The model architecture to load checkpoints into
        dataloader: Evaluation dataloader

    Returns:
        steps: List of training steps
        importance_history: Tensor of shape (n_checkpoints, n_layers, n_heads)
    """
    if device is None:
        device = next(model.parameters()).device

    steps = []
    importance_history = []

    for path in checkpoint_paths:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except:
            ckpt = torch.load(path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt['model_state'])
        step = ckpt.get('step', -1)

        importance = compute_head_importance(model, dataloader, device)

        steps.append(step)
        importance_history.append(importance)

    return steps, torch.stack(importance_history)

def analyze_mlp_neurons(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                       device: torch.device = None) -> Dict[str, torch.Tensor]:
    """
    Identify MLP neurons that activate specifically for task-relevant inputs.
    Uses forward hooks to capture FFN activations.

    Args:
        model: ModularArithmeticTransformer model
        dataloader: Dataloader with modular arithmetic tasks

    Returns:
        Dict with layer names as keys and neuron activation stats as values.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    activations = defaultdict(list)

    def get_activation_hook(name):
        def hook(model, input, output):
            # output of linear1 in FFN is typically before activation, but let's capture the activated state
            # If it's a Sequential or we attach to the activation function directly, output is what we want.
            # Assuming standard PyTorch TransformerEncoderLayer where FFN is linear1 -> act -> linear2
            # We'll attach to linear1 and apply activation manually, or attach to linear2 input.

            # Since standard PyTorch layer doesn't easily expose the activated hidden state,
            # we capture linear1 output and apply GELU (the default in model.py)
            act = F.gelu(output).detach().cpu()
            activations[name].append(act)
        return hook

    hooks = []
    for i, layer in enumerate(model.transformer.layers):
        hook = layer.linear1.register_forward_hook(get_activation_hook(f"layer_{i}_ffn"))
        hooks.append(hook)

    # Run data through
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Aggregate and analyze
    results = {}
    for name, acts in activations.items():
        # Stack all batches
        stacked = torch.cat(acts, dim=0) # shape (total_samples, seq_len, d_ff)

        # Compute statistics across samples (mean activation, variance, kurtosis)
        # Pool across sequence length for simplicity or analyze separately
        pooled = stacked.mean(dim=1) # shape (total_samples, d_ff)

        mean_act = pooled.mean(dim=0)
        var_act = pooled.var(dim=0)

        # Kurtosis to measure sparsity/selectivity (Polysemanticity proxy)
        # kurtosis = E[(X-mu)^4] / sigma^4 - 3
        centered = pooled - mean_act
        m4 = (centered ** 4).mean(dim=0)
        m2 = (centered ** 2).mean(dim=0)
        kurtosis = m4 / (m2 ** 2 + 1e-8) - 3.0

        results[name] = {
            'mean': mean_act,
            'var': var_act,
            'kurtosis': kurtosis
        }

    return results

def detect_circuit_emergence(attention_scores_history: List[torch.Tensor], threshold: float = 0.5) -> int:
    """
    Detect when functional circuits stabilize based on inter-head communication or attention score stability.

    Args:
        attention_scores_history: List of attention matrices or importance scores over time
        threshold: Variance or change threshold to define 'stability'

    Returns:
        Integer representing the step index where stabilization occurs, or -1 if not found.
    """
    if len(attention_scores_history) < 2:
        return -1

    # Compute differences between consecutive steps
    diffs = []
    for i in range(1, len(attention_scores_history)):
        diff = torch.norm(attention_scores_history[i] - attention_scores_history[i-1]).item()
        diffs.append(diff)

    # Find where the rate of change drops below threshold for a sustained period (e.g., 3 steps)
    sustained = 3
    for i in range(len(diffs) - sustained + 1):
        window = diffs[i:i+sustained]
        if all(d < threshold for d in window):
            return i + 1 # Return index in history

    return -1
