import torch
import torch.nn.functional as F
import numpy as np


def identify_important_circuits(model, data, targets, criterion=F.cross_entropy):
    """
    Use activation patching / ablation to find critical attention heads.
    Since we can't easily hook into nn.TransformerEncoderLayer during a forward pass
    without modifying the model source, we manually simulate the forward pass for 1 layer,
    zeroing out the contribution of specific heads.

    Args:
        model: ModularArithmeticTransformer
        data: Input tensor (batch_size, seq_len)
        targets: Target tensor (batch_size)
        criterion: Loss function

    Returns:
        importance_scores: List of importance scores for each head
    """
    model.eval()

    # 1. Get baseline loss
    with torch.no_grad():
        baseline_logits = model(data)
        baseline_loss = criterion(baseline_logits, targets).item()

    batch_size = data.shape[0]
    seq_len = data.shape[1]
    d_model = model.d_model
    n_heads = model.n_heads
    head_dim = d_model // n_heads

    importance_scores = []

    with torch.no_grad():
        # Input embeddings
        tok = model.token_embed(data)
        positions = torch.arange(seq_len, device=data.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        x = tok + pos

        # Self attention parameters
        attn_layer = model.transformer.layers[0].self_attn
        in_proj_weight = attn_layer.in_proj_weight
        in_proj_bias = attn_layer.in_proj_bias
        out_proj_weight = attn_layer.out_proj.weight
        out_proj_bias = attn_layer.out_proj.bias

        # Project x to Q, K, V
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Compute head outputs: (batch, n_heads, seq_len, head_dim)
        head_outputs = torch.matmul(attn_weights, v)

        for head_idx in range(n_heads):
            # Create ablated head outputs
            ablated_head_outputs = head_outputs.clone()
            # Zero out the specific head
            ablated_head_outputs[:, head_idx, :, :] = 0.0

            # Recombine heads and apply out_proj
            # (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, d_model)
            concat_out = ablated_head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

            # Apply out_proj
            attn_out = F.linear(concat_out, out_proj_weight, out_proj_bias)

            # Add residual connection and layer norm (first LN in standard Post-LN, or in this case we're mirroring nn.TransformerEncoderLayer)
            # PyTorch's default TransformerEncoderLayer applies LayerNorm AFTER the residual connection (Post-LN) or BEFORE (Pre-LN).
            # Default is Post-LN: x = LayerNorm(x + Sublayer(x))
            # However, PyTorch's forward pass is:
            # src2 = self.self_attn(src, src, src...)[0]
            # src = src + self.dropout1(src2)
            # src = self.norm1(src)
            # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            # src = src + self.dropout2(src2)
            # src = self.norm2(src)

            layer = model.transformer.layers[0]

            # Simulate the rest of the layer
            src = x + attn_out
            src = layer.norm1(src)

            src2 = layer.linear2(F.gelu(layer.linear1(src)))
            src = src + src2
            src = layer.norm2(src)

            # Model final layernorm
            h = model.ln(src)

            # Pool across positions (mean) and predict
            h = h.mean(dim=1)
            ablated_logits = model.output_head(h)

            ablated_loss = criterion(ablated_logits, targets).item()

            # Importance is how much loss increased when head was ablated
            importance = ablated_loss - baseline_loss
            importance_scores.append(importance)

    return importance_scores


def compute_circuit_importance(model, baseline_data, patched_data):
    """
    Alternative API for circuit importance, e.g. for causal scrubbing.
    Not strictly required given identify_important_circuits, but provided for completeness.
    """
    pass


def track_circuit_formation_across_collapse(models_dict, data, targets):
    """
    Track circuit importance across different checkpoints/conditions.

    Args:
        models_dict: Dict mapping condition_name -> list of models (checkpoints)
        data: Input tensor
        targets: Target tensor

    Returns:
        results: Dict mapping condition_name -> list of importance scores (one list per checkpoint)
    """
    results = {}
    for condition_name, models in models_dict.items():
        condition_results = []
        for model in models:
            scores = identify_important_circuits(model, data, targets)
            condition_results.append(scores)
        results[condition_name] = condition_results

    return results
