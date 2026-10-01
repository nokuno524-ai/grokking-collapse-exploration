import torch
import torch.nn as nn
from typing import Dict, List, Literal

def ablate_head(model: nn.Module, head_idx: int, method: Literal["zero", "mean"] = "zero") -> None:
    """
    Ablate a specific attention head in the first layer's MultiheadAttention.

    Args:
        model: ModularArithmeticTransformer model.
        head_idx: Index of the head to ablate.
        method: 'zero' or 'mean'. Zero sets weights to 0. Mean sets them to their mean.
    """
    # Find the MHA module
    mha = None
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            mha = module
            break

    if mha is None:
        raise ValueError("No MultiheadAttention found in model.")

    d_model = mha.embed_dim
    n_heads = mha.num_heads
    d_head = d_model // n_heads

    start_idx = head_idx * d_head
    end_idx = (head_idx + 1) * d_head

    # In PyTorch MultiheadAttention, in_proj_weight is shape (3 * d_model, d_model)
    # The layout is [Q, K, V]
    if mha.in_proj_weight is not None:
        with torch.no_grad():
            for offset in [0, d_model, 2 * d_model]:
                if method == "zero":
                    mha.in_proj_weight[offset + start_idx : offset + end_idx, :] = 0.0
                elif method == "mean":
                    mean_val = mha.in_proj_weight[offset + start_idx : offset + end_idx, :].mean()
                    mha.in_proj_weight[offset + start_idx : offset + end_idx, :] = mean_val

    if mha.in_proj_bias is not None:
        with torch.no_grad():
            for offset in [0, d_model, 2 * d_model]:
                if method == "zero":
                    mha.in_proj_bias[offset + start_idx : offset + end_idx] = 0.0
                elif method == "mean":
                    mean_val = mha.in_proj_bias[offset + start_idx : offset + end_idx].mean()
                    mha.in_proj_bias[offset + start_idx : offset + end_idx] = mean_val

    # out_proj is shape (d_model, d_model)
    if hasattr(mha, 'out_proj') and mha.out_proj.weight is not None:
        with torch.no_grad():
            if method == "zero":
                mha.out_proj.weight[:, start_idx : end_idx] = 0.0
            elif method == "mean":
                mean_val = mha.out_proj.weight[:, start_idx : end_idx].mean()
                mha.out_proj.weight[:, start_idx : end_idx] = mean_val

def evaluate_ablation(model: nn.Module, dataloader: torch.utils.data.DataLoader, head_classes: List[str]) -> Dict[str, float]:
    """
    Quantify importance of each head class by ablating all heads in that class.

    Args:
        model: ModularArithmeticTransformer model.
        dataloader: DataLoader for evaluation.
        head_classes: List of class labels for each head.

    Returns:
        Dict mapping class name to relative accuracy drop.
    """
    device = next(model.parameters()).device
    model.eval()

    def evaluate(m):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = m(inputs)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += inputs.shape[0]
        return correct / total if total > 0 else 0.0

    # Baseline accuracy
    base_acc = evaluate(model)

    # Ablate classes
    results = {}
    classes = set(head_classes)

    import copy

    for cls in classes:
        ablated_model = copy.deepcopy(model)
        for idx, head_cls in enumerate(head_classes):
            if head_cls == cls:
                ablate_head(ablated_model, idx, method="zero")

        ablated_acc = evaluate(ablated_model)
        # Store absolute accuracy, or we could store relative drop
        results[f"{cls}_ablation_acc"] = ablated_acc

    results["baseline_acc"] = base_acc

    return results
