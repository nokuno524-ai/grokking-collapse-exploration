import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
import json
import numpy as np

def ablate_head(model: nn.Module, layer_idx: int, head_idx: int, ablation_type: str = "zero") -> Callable:
    """
    Temporarily modify out_proj.weight to ablate a specific head.
    Standard PyTorch nn.TransformerEncoderLayer doesn't expose per-head outputs natively.
    Returns a cleanup function to restore the weights.
    """
    # Find the attention layer
    layer = model.transformer.layers[layer_idx].self_attn
    out_proj = layer.out_proj

    d_model = out_proj.weight.shape[0]
    n_heads = layer.num_heads
    head_dim = d_model // n_heads

    # Save original weight
    orig_weight = out_proj.weight.data.clone()

    # Modify weight
    with torch.no_grad():
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim

        if ablation_type == "zero":
            out_proj.weight.data[:, start_idx:end_idx] = 0
        elif ablation_type == "mean":
            # Compute mean across the head dimension for each output feature
            mean_val = out_proj.weight.data[:, start_idx:end_idx].mean(dim=1, keepdim=True)
            out_proj.weight.data[:, start_idx:end_idx] = mean_val
        else:
            raise ValueError(f"Unknown ablation_type: {ablation_type}")

    def cleanup():
        out_proj.weight.data.copy_(orig_weight)

    return cleanup

def compute_activation_patches(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_idx: int,
    device: torch.device,
    ablation_type: str = "zero"
) -> Dict[int, float]:
    """
    Compute causal importance of each head in a given layer via ablation.
    Returns dictionary mapping head index to performance drop (importance).
    """
    model.eval()

    def evaluate():
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                total_loss += loss.item() * inputs.shape[0]
                total += inputs.shape[0]
        return total_loss / total

    base_loss = evaluate()

    layer = model.transformer.layers[layer_idx].self_attn
    n_heads = layer.num_heads

    importance = {}
    for head_idx in range(n_heads):
        cleanup = ablate_head(model, layer_idx, head_idx, ablation_type=ablation_type)
        ablated_loss = evaluate()
        cleanup()
        importance[head_idx] = float(ablated_loss - base_loss)

    return importance

def identify_grokking_heads(
    pre_grok_model: nn.Module,
    post_grok_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_path: Optional[str] = None
) -> Dict[str, Dict[int, float]]:
    """
    Identify heads involved in the grokking transition.
    Compares causal importance pre-grokking vs post-grokking.
    """
    n_layers = len(pre_grok_model.transformer.layers)

    results = {}
    for layer_idx in range(n_layers):
        pre_importance = compute_activation_patches(pre_grok_model, dataloader, layer_idx, device)
        post_importance = compute_activation_patches(post_grok_model, dataloader, layer_idx, device)

        diff_importance = {}
        for head_idx in pre_importance.keys():
            diff_importance[head_idx] = post_importance[head_idx] - pre_importance[head_idx]

        results[f"layer_{layer_idx}_pre"] = pre_importance
        results[f"layer_{layer_idx}_post"] = post_importance
        results[f"layer_{layer_idx}_diff"] = diff_importance

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
