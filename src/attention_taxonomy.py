import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np

def extract_attention_weights(model: nn.Module, inputs: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract attention weights from all layers of ModularArithmeticTransformer.
    As per notes: pass sum of token and pos embed to layer.self_attn with need_weights=True.
    Returns a list of tensors of shape (batch, n_heads, seq_len, seq_len).
    """
    model.eval()
    with torch.no_grad():
        batch_size = inputs.shape[0]
        tok = model.token_embed(inputs)
        positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)

        h = tok + pos

        attn_weights = []
        for layer in model.transformer.layers:
            # Need to handle batch_first if present, but standard nn.MultiheadAttention expects (seq_len, batch, dim) unless batch_first=True
            batch_first = getattr(layer.self_attn, 'batch_first', False)

            if batch_first:
                x = h
            else:
                x = h.transpose(0, 1)

            # self_attn(query, key, value, need_weights=True, average_attn_weights=False)
            attn_output, attn_weight_tensor = layer.self_attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False
            )

            # attn_weight_tensor is (batch, n_heads, seq_len, seq_len)
            attn_weights.append(attn_weight_tensor)

            h = layer(h)

    return attn_weights

def detect_previous_token_heads(attn_weights: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """
    Identifies heads that primarily attend to the previous token.
    attn_weights shape: (batch, n_heads, seq_len, seq_len)
    """
    batch_size, n_heads, seq_len, _ = attn_weights.shape
    if seq_len < 2:
        return []

    prev_token_heads = []

    # Check attention from token i to i-1
    # We average over all batches, and all relevant positions (i >= 1)
    for head_idx in range(n_heads):
        head_attn = attn_weights[:, head_idx, :, :] # (batch, seq_len, seq_len)

        # Calculate average attention to the previous token
        score = 0
        for pos in range(1, seq_len):
            score += head_attn[:, pos, pos-1].mean().item()
        score /= (seq_len - 1)

        if score > threshold:
            prev_token_heads.append(head_idx)

    return prev_token_heads

def detect_duplicate_token_heads(attn_weights: torch.Tensor, inputs: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """
    Identifies heads that primarily attend to duplicate tokens in the sequence.
    """
    batch_size, n_heads, seq_len, _ = attn_weights.shape
    duplicate_token_heads = []

    for head_idx in range(n_heads):
        head_attn = attn_weights[:, head_idx, :, :]
        score = 0
        count = 0

        for b in range(batch_size):
            for i in range(1, seq_len):
                for j in range(i):
                    if inputs[b, i] == inputs[b, j]:
                        score += head_attn[b, i, j].item()
                        count += 1

        if count > 0 and (score / count) > threshold:
            duplicate_token_heads.append(head_idx)

    return duplicate_token_heads

def detect_induction_heads(attn_weights: torch.Tensor, inputs: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """
    Identifies induction heads (attending to token following previous occurrence).
    Requires seq_len > 2 to be meaningful.
    """
    batch_size, n_heads, seq_len, _ = attn_weights.shape
    induction_heads = []

    for head_idx in range(n_heads):
        head_attn = attn_weights[:, head_idx, :, :]
        score = 0
        count = 0

        for b in range(batch_size):
            for i in range(2, seq_len):
                for j in range(i - 1):
                    # If current token matches a previous token (induction prefix)
                    if inputs[b, i-1] == inputs[b, j]:
                        # Check attention from current token to the token following the prefix
                        score += head_attn[b, i, j+1].item()
                        count += 1

        if count > 0 and (score / count) > threshold:
            induction_heads.append(head_idx)

    return induction_heads

def plot_taxonomy_heatmap(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_path: str
):
    """
    Extracts attention weights, classifies heads, and visualizes the taxonomy as a heatmap.
    """
    model.eval()

    # Collect attention weights for a few batches
    all_attn_weights = []
    all_inputs = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= 5: # Limit to 5 batches for taxonomy
                break
            inputs = inputs.to(device)
            attn_weights = extract_attention_weights(model, inputs)
            all_attn_weights.append(attn_weights)
            all_inputs.append(inputs)

    if not all_attn_weights:
        return

    n_layers = len(all_attn_weights[0])
    n_heads = all_attn_weights[0][0].shape[1]

    taxonomy_matrix = np.zeros((n_layers * n_heads, 3)) # 3 classes: Prev, Dup, Ind

    # Average across batches
    for layer_idx in range(n_layers):
        layer_attn = torch.cat([aw[layer_idx] for aw in all_attn_weights], dim=0)
        inputs_concat = torch.cat(all_inputs, dim=0)

        prev_heads = detect_previous_token_heads(layer_attn, threshold=0.3)
        dup_heads = detect_duplicate_token_heads(layer_attn, inputs_concat, threshold=0.3)
        ind_heads = detect_induction_heads(layer_attn, inputs_concat, threshold=0.3)

        for head_idx in range(n_heads):
            row_idx = layer_idx * n_heads + head_idx
            if head_idx in prev_heads:
                taxonomy_matrix[row_idx, 0] = 1
            if head_idx in dup_heads:
                taxonomy_matrix[row_idx, 1] = 1
            if head_idx in ind_heads:
                taxonomy_matrix[row_idx, 2] = 1

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        taxonomy_matrix,
        cmap="Blues",
        cbar=False,
        xticklabels=["Previous Token", "Duplicate Token", "Induction"],
        yticklabels=[f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    )
    plt.title("Attention Head Taxonomy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
