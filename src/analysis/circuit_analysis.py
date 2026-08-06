import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable

def activation_patch(model: nn.Module, clean_input: torch.Tensor, corrupted_input: torch.Tensor, layer_range: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform activation patching. Runs clean_input to cache activations, then runs corrupted_input
    but patches in the clean activations at the specified layers.

    Args:
        model: The ModularArithmeticTransformer model.
        clean_input: Clean input tensor (batch, seq_len).
        corrupted_input: Corrupted input tensor (batch, seq_len).
        layer_range: List of layer indices to patch (for 1-layer model, usually [0]).

    Returns:
        clean_logits, patched_logits
    """
    clean_cache = {}
    hooks = []

    def get_cache_hook(layer_idx):
        def hook(module, args, output):
            clean_cache[layer_idx] = output.detach()
            return output
        return hook

    def get_patch_hook(layer_idx):
        def hook(module, args, output):
            return clean_cache[layer_idx]
        return hook

    # Register cache hooks
    for layer_idx in layer_range:
        hook_handle = model.transformer.layers[layer_idx].register_forward_hook(get_cache_hook(layer_idx))
        hooks.append(hook_handle)

    # Forward clean
    with torch.no_grad():
        clean_logits = model(clean_input)

    # Remove cache hooks
    for hook_handle in hooks:
        hook_handle.remove()
    hooks.clear()

    # Register patch hooks
    for layer_idx in layer_range:
        hook_handle = model.transformer.layers[layer_idx].register_forward_hook(get_patch_hook(layer_idx))
        hooks.append(hook_handle)

    # Forward corrupted
    with torch.no_grad():
        patched_logits = model(corrupted_input)

    # Remove patch hooks
    for hook_handle in hooks:
        hook_handle.remove()

    return clean_logits, patched_logits


def discover_circuits(model: nn.Module, dataset: torch.utils.data.Dataset) -> Dict[str, float]:
    """
    Identify which components recover original prediction using activation patching.
    A simple demonstration for the 1-layer ModularArithmeticTransformer.

    Args:
        model: The model.
        dataset: A dataset providing (inputs, targets).

    Returns:
        Dictionary mapping component names to their importance/recovery score.
    """
    # Sample a small batch for discovery
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    clean_input, clean_target = next(iter(dataloader))

    # Create corrupted input by rolling the tokens
    corrupted_input = torch.roll(clean_input, shifts=1, dims=0)

    # Baseline: corrupted loss
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        clean_logits = model(clean_input)
        corrupted_logits = model(corrupted_input)

        clean_loss = criterion(clean_logits, clean_target).item()
        corrupted_loss = criterion(corrupted_logits, clean_target).item()

    scores = {}

    # Patch layer 0
    _, patched_logits = activation_patch(model, clean_input, corrupted_input, [0])
    patched_loss = criterion(patched_logits, clean_target).item()

    # Score is how much of the gap is closed (1.0 = fully recovered, 0.0 = no recovery)
    if corrupted_loss - clean_loss > 1e-6:
        recovery = (corrupted_loss - patched_loss) / (corrupted_loss - clean_loss)
    else:
        recovery = 0.0

    scores['layer_0'] = recovery
    return scores


def head_importance_scores(model: nn.Module, dataset: torch.utils.data.Dataset) -> torch.Tensor:
    """
    Compute importance score for each attention head using grad * activation.

    Args:
        model: The ModularArithmeticTransformer model.
        dataset: A dataset providing (inputs, targets).

    Returns:
        Tensor of shape (n_heads,) with importance scores.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    inputs, targets = next(iter(dataloader))

    n_heads = model.n_heads
    d_model = model.d_model
    head_dim = d_model // n_heads

    # Enable gradients for the parameters and inputs for attribution
    model.train()
    model.zero_grad()

    # We will use the gradients of the out_proj weight to estimate head importance
    # as an alternative to hooks which can be fragile with nn.TransformerEncoderLayer

    logits = model(inputs)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    out_proj_weight = model.transformer.layers[0].self_attn.out_proj.weight
    grad = out_proj_weight.grad # (d_model, d_model)
    weight = out_proj_weight.data

    if grad is None:
        return torch.zeros(n_heads)

    # Salience = |grad * weight|
    # out_proj receives concatenated head outputs. The input dimension (dim 1) is grouped by head.
    salience = (grad * weight).abs().mean(dim=0) # (d_model,)

    importance = salience.view(n_heads, head_dim).sum(dim=1)
    if importance.sum() > 0:
        importance = importance / importance.sum()

    return importance
