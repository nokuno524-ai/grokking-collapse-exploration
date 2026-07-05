import torch
import torch.nn as nn
from typing import Dict, Tuple

def ablate_attention_head(model: nn.Module, layer_idx: int, head_idx: int) -> Tuple[nn.Module, torch.Tensor]:
    """
    Temporarily zero out the contribution of a specific attention head
    by modifying the out_proj.weight matrix of the specified layer.

    Returns:
        The modified model (in-place modified but returned for convenience),
        and the original weight tensor to restore it later.
    """
    layer = model.transformer.layers[layer_idx]
    multihead_attn = layer.self_attn

    d_model = multihead_attn.embed_dim
    n_heads = multihead_attn.num_heads
    head_dim = d_model // n_heads

    # Save original weights
    out_proj_weight = multihead_attn.out_proj.weight.data.clone()

    # Zero out the specific head's columns in out_proj.weight
    # out_proj weight shape is (d_model, d_model).
    # The columns correspond to the concatenated outputs of the heads.
    start_idx = head_idx * head_dim
    end_idx = start_idx + head_dim

    multihead_attn.out_proj.weight.data[:, start_idx:end_idx] = 0.0

    return model, out_proj_weight

def restore_attention_head(model: nn.Module, layer_idx: int, original_weight: torch.Tensor):
    """
    Restore the original out_proj.weight matrix.
    """
    layer = model.transformer.layers[layer_idx]
    layer.self_attn.out_proj.weight.data = original_weight

def evaluate_head_importance(model: nn.Module, dataloader, device, loss_fn) -> Dict[Tuple[int, int], float]:
    """
    Evaluate importance of each head by ablating it and measuring the increase in loss.
    """
    model.eval()

    # Base loss
    base_loss = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            base_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
    base_loss /= total

    importance_scores = {}

    n_layers = len(model.transformer.layers)
    n_heads = model.transformer.layers[0].self_attn.num_heads

    for l in range(n_layers):
        for h in range(n_heads):
            # Ablate
            _, orig_weight = ablate_attention_head(model, l, h)

            # Evaluate
            ablated_loss = 0.0
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits = model(inputs)
                    loss = loss_fn(logits, targets)
                    ablated_loss += loss.item() * inputs.size(0)
            ablated_loss /= total

            # Score is increase in loss (higher means more important)
            importance_scores[(l, h)] = ablated_loss - base_loss

            # Restore
            restore_attention_head(model, l, orig_weight)

    return importance_scores

if __name__ == "__main__":
    from src.model import ModularArithmeticTransformer
    import torch.nn.functional as F

    model = ModularArithmeticTransformer()
    x = torch.randint(0, 59, (4, 2))
    y = torch.randint(0, 59, (4,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    scores = evaluate_head_importance(model, loader, torch.device("cpu"), F.cross_entropy)
    print("Head importance scores:")
    for k, v in scores.items():
        print(f"Layer {k[0]}, Head {k[1]}: {v:.4f}")
