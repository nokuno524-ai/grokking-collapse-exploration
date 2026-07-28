import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

def compute_attention_rank(model, inputs: torch.Tensor) -> Dict[str, float]:
    """
    Compute effective rank of attention mechanisms at each layer.
    Extracts Q, K matrices and computes rank via SVD.

    Args:
        model: ModularArithmeticTransformer
        inputs: Input tensor (batch, 2)

    Returns:
        Dictionary of effective ranks for each head/layer.
    """
    model.eval()
    ranks = {}

    with torch.no_grad():
        seq_len = inputs.shape[1]
        # Get embeddings
        tok = model.token_embed(inputs)
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        # We need to extract the in_proj_weight from the transformer encoder layer
        # Since it's a 1-layer transformer, we access the first layer
        layer = model.transformer.layers[0]

        # QKV projection weights
        in_proj_weight = layer.self_attn.in_proj_weight
        in_proj_bias = layer.self_attn.in_proj_bias

        d_model = model.d_model
        n_heads = model.n_heads
        head_dim = d_model // n_heads

        # Project
        qkv = F.linear(h, in_proj_weight, in_proj_bias) # (batch, seq_len, 3*d_model)

        # Reshape to separate Q, K, V
        qkv = qkv.reshape(inputs.shape[0], seq_len, 3, n_heads, head_dim)
        q = qkv[:, :, 0, :, :] # (batch, seq, heads, dim)
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        # Compute attention scores
        # (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1) # (batch, heads, seq, seq)

        # Compute effective rank of the attention matrix per head
        # Average across batch
        avg_attn = attn.mean(dim=0) # (heads, seq, seq)

        for head in range(n_heads):
            # Compute SVD of the attention matrix for this head
            matrix = avg_attn[head]
            try:
                s = torch.linalg.svdvals(matrix)
                # normalize
                s = s / (s.sum() + 1e-10)
                entropy = -(s * torch.log(s + 1e-10)).sum()
                rank = torch.exp(entropy).item()
            except Exception:
                rank = float('nan')

            ranks[f"layer0_head{head}_rank"] = rank

    return ranks

def compute_participation_ratio(activations: torch.Tensor) -> float:
    """
    Compute the participation ratio (effective dimensionality) of representations.

    Args:
        activations: Tensor of shape (..., d_model)

    Returns:
        Participation ratio.
    """
    if len(activations.shape) > 2:
        activations = activations.reshape(-1, activations.shape[-1])

    # Center activations
    activations = activations - activations.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    cov = torch.matmul(activations.T, activations) / (activations.shape[0] - 1)

    try:
        eigenvalues = torch.linalg.eigvalsh(cov)
        # Participation ratio: (sum(eigenvalues))^2 / sum(eigenvalues^2)
        pr = (eigenvalues.sum() ** 2) / (torch.sum(eigenvalues ** 2) + 1e-10)
        return pr.item()
    except Exception:
        return float('nan')

def compute_information_flow(model, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute mutual information proxies between inputs/outputs.
    Uses CKA (Centered Kernel Alignment) as a proxy for information flow.

    Args:
        model: ModularArithmeticTransformer
        inputs: Input tensor (batch, 2)
        targets: Target tensor (batch,)

    Returns:
        Dictionary of information flow metrics.
    """
    model.eval()

    def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
        if len(x.shape) > 2: x = x.reshape(x.shape[0], -1)
        if len(y.shape) > 2: y = y.reshape(y.shape[0], -1)

        x_c = x - x.mean(dim=0, keepdim=True)
        y_c = y - y.mean(dim=0, keepdim=True)

        dot_prod = torch.norm(torch.matmul(x_c.T, y_c)) ** 2
        norm_x = torch.norm(torch.matmul(x_c.T, x_c))
        norm_y = torch.norm(torch.matmul(y_c.T, y_c))

        return (dot_prod / (norm_x * norm_y + 1e-10)).item()

    metrics = {}
    with torch.no_grad():
        seq_len = inputs.shape[1]
        tok = model.token_embed(inputs)
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
        pos = model.pos_embed(positions)
        h_in = tok + pos

        h_out = model.transformer(h_in)
        h_out = model.ln(h_out)

        pooled = h_out.mean(dim=1)
        logits = model.output_head(pooled)

        # One-hot encode targets for comparison
        targets_oh = F.one_hot(targets, num_classes=model.prime).float()

        metrics["cka_input_output_layer"] = linear_cka(h_in, h_out)
        metrics["cka_pooled_logits"] = linear_cka(pooled, logits)
        metrics["cka_logits_targets"] = linear_cka(logits, targets_oh)
        metrics["pr_input"] = compute_participation_ratio(h_in)
        metrics["pr_output"] = compute_participation_ratio(h_out)

    return metrics
