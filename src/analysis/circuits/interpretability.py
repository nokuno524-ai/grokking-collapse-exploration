import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

def compute_head_importance(model: nn.Module, dataloader: DataLoader, device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
    """
    Compute gradient-based head importance scoring (Integrated Gradients / Attributions).
    Uses the gradients of self_attn.out_proj.weight after a backward pass.
    """
    model.eval()
    model.to(device)

    # Store initial weights to restore later if necessary

    # We will accumulate the absolute gradients over the dataloader
    attributions = None
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.zero_grad()

        # We need to manually access the attention output projection to get gradients
        # ModularArithmeticTransformer uses nn.TransformerEncoderLayer
        # We access the first layer's self attention out_proj
        try:
            out_proj = model.transformer.layers[0].self_attn.out_proj
        except AttributeError:
            continue

        # Get logits
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        if out_proj.weight.grad is not None:
            grad = out_proj.weight.grad.abs().detach()
            if attributions is None:
                attributions = grad.sum(dim=0)
            else:
                attributions += grad.sum(dim=0)

        total_samples += inputs.size(0)

    if attributions is None or total_samples == 0:
        return {}

    # Average attribution per feature
    attributions = attributions / total_samples

    # Calculate per-head attribution
    d_model = attributions.shape[0]
    n_heads = getattr(model, 'n_heads', 4) # fallback to 4
    head_dim = d_model // n_heads

    head_importance = {}
    for i in range(n_heads):
        head_importance[f"head_{i}"] = attributions[i * head_dim:(i + 1) * head_dim].mean().item()

    return head_importance
