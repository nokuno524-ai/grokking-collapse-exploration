import torch
from src.model import ModularArithmeticTransformer

def compute_head_class_usage(model: ModularArithmeticTransformer, dataloader) -> torch.Tensor:
    """
    Computes the average attention weights across the dataset to measure head-class usage.
    Returns: Tensor of shape (n_layers, n_heads, seq_len, seq_len)
    """
    device = next(model.parameters()).device
    model.eval()
    all_attn = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            attn = model.get_attention_weights(inputs)  # (layers, batch, heads, seq, seq)
            all_attn.append(attn.mean(dim=1)) # average over batch

    if not all_attn:
        return torch.zeros(0)

    return torch.stack(all_attn).mean(dim=0)
