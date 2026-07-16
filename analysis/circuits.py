import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

def extract_attention_patterns(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extracts attention head patterns using layer.self_attn.

    Args:
        model: ModularArithmeticTransformer model.
        x: Input tensor of shape (batch, 2).

    Returns:
        Attention weights of shape (batch, n_heads, seq_len, seq_len).
    """
    # Token embeddings
    tok = model.token_embed(x)  # (batch, 2, d_model)

    # Positional embeddings
    positions = torch.arange(2, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    pos = model.pos_embed(positions)  # (batch, 2, d_model)

    # Combine
    h = tok + pos  # (batch, 2, d_model)

    # Extract attention from the first layer
    layer = model.transformer.layers[0]

    # self_attn expects query, key, value
    # For PyTorch nn.MultiheadAttention, inputs are (seq_len, batch, embed_dim) if batch_first=False
    # but the model initializes with batch_first=True
    attn_output, attn_weights = layer.self_attn(
        h, h, h,
        need_weights=True,
        average_attn_weights=False
    )

    return attn_weights

def track_circuit_formation(checkpoints: List[Dict], eval_data: torch.Tensor) -> List[torch.Tensor]:
    """
    Tracks circuit formation over training by extracting attention patterns at each checkpoint.

    Args:
        checkpoints: List of model state dicts or checkpoint objects.
        eval_data: Data to evaluate attention patterns on.

    Returns:
        List of attention weights for each checkpoint.
    """
    from src.model import ModularArithmeticTransformer

    patterns = []

    for ckpt in checkpoints:
        # Reconstruct model
        model = ModularArithmeticTransformer()
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        with torch.no_grad():
            weights = extract_attention_patterns(model, eval_data)
            patterns.append(weights)

    return patterns

def compare_circuit_structures(patterns_pure: torch.Tensor, patterns_collapsed: torch.Tensor) -> Dict[str, float]:
    """
    Compares the circuit structures between pure and collapsed models.

    Args:
        patterns_pure: Attention weights of pure model.
        patterns_collapsed: Attention weights of collapsed model.

    Returns:
        Dictionary containing comparison metrics (e.g., L2 difference).
    """
    diff = torch.norm(patterns_pure - patterns_collapsed, p=2)
    cos_sim = torch.nn.functional.cosine_similarity(
        patterns_pure.flatten(), patterns_collapsed.flatten(), dim=0
    )

    return {
        "l2_diff": diff.item(),
        "cosine_similarity": cos_sim.item()
    }

def identify_grokking_circuits(pre_grok_patterns: torch.Tensor, post_grok_patterns: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """
    Identifies heads that specialize after grokking onset.
    A head is considered a grokking circuit if its attention pattern changes significantly.

    Args:
        pre_grok_patterns: Attention weights pre-grokking (batch, n_heads, seq_len, seq_len).
        post_grok_patterns: Attention weights post-grokking (batch, n_heads, seq_len, seq_len).
        threshold: L2 norm difference threshold for identifying specialization.

    Returns:
        List of head indices that specialized.
    """
    # Average across batch
    pre_avg = pre_grok_patterns.mean(dim=0)
    post_avg = post_grok_patterns.mean(dim=0)

    specialized_heads = []
    n_heads = pre_avg.shape[0]

    for head_idx in range(n_heads):
        diff = torch.norm(pre_avg[head_idx] - post_avg[head_idx], p=2).item()
        if diff > threshold:
            specialized_heads.append(head_idx)

    return specialized_heads
