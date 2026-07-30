import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
import copy

def get_logit_attribution(model: nn.Module, inputs: torch.Tensor, target_idx: int) -> Dict[str, float]:
    """
    Decompose model output (logits) into contributions from the embedding, positions, and attention output.
    Since this is a 1-layer transformer (ModularArithmeticTransformer), we can track the residual stream.
    """
    model.eval()

    with torch.no_grad():
        # Get token embeddings
        tok = model.token_embed(inputs) # (batch, 2, d_model)

        # Get positional embeddings
        positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
        pos = model.pos_embed(positions) # (batch, 2, d_model)

        h_0 = tok + pos # Initial residual stream

        # Pass through the transformer encoder layer
        # The layer contains MultiheadAttention and MLP, wrapped with LayerNorms and residuals
        encoder_layer = model.transformer.layers[0]

        # First norm and attention
        h_norm1 = encoder_layer.norm1(h_0)
        attn_out, attn_weights = encoder_layer.self_attn(h_norm1, h_norm1, h_norm1, need_weights=True)
        h_1 = h_0 + encoder_layer.dropout1(attn_out)

        # Second norm and MLP
        h_norm2 = encoder_layer.norm2(h_1)
        mlp_out = encoder_layer.linear2(encoder_layer.dropout(encoder_layer.activation(encoder_layer.linear1(h_norm2))))
        h_2 = h_1 + encoder_layer.dropout2(mlp_out)

        # Final layer norm (part of the model)
        h_final = model.ln(h_2)
        h_final_mean = h_final.mean(dim=1)

        # To strictly do linear attribution, we'd have to approximate LayerNorm.
        # But we can look at the un-normed dot products with the output weights for the target class.
        W_U = model.output_head.weight[target_idx] # (d_model,)
        b_U = model.output_head.bias[target_idx] if model.output_head.bias is not None else 0.0

        # We will attribute based on h_2 before the final layernorm, scaled approximately by the layernorm scale.
        # Actually, let's just trace the components projected by W_U directly.
        # We need to average over sequence length first (mean pooling)

        # 1. Direct path (Embedding)
        embed_contrib = (h_0.mean(dim=1) @ W_U).item()

        # 2. Attention path
        attn_contrib = (attn_out.mean(dim=1) @ W_U).item()

        # 3. MLP path
        mlp_contrib = (mlp_out.mean(dim=1) @ W_U).item()

        total_logit = model(inputs)[0, target_idx].item()

        return {
            "embed_contrib": embed_contrib,
            "attn_contrib": attn_contrib,
            "mlp_contrib": mlp_contrib,
            "total_logit": total_logit,
            "bias": b_U.item() if isinstance(b_U, torch.Tensor) else b_U
        }

def activation_patching(clean_model: nn.Module, corrupt_model: nn.Module, clean_input: torch.Tensor, patch_layer: str) -> torch.Tensor:
    """
    Patch activations from a clean (grokked) model into a corrupt (collapsed) model to see if it recovers performance.
    patch_layer can be 'embed', 'attn_out', or 'mlp_out'.
    Returns patched logits from the corrupt model.
    """
    clean_model.eval()
    corrupt_model.eval()

    # Run clean model to collect activations
    clean_acts = {}
    with torch.no_grad():
        tok = clean_model.token_embed(clean_input)
        positions = torch.arange(2, device=clean_input.device).unsqueeze(0).expand(clean_input.shape[0], -1)
        pos = clean_model.pos_embed(positions)
        h_0 = tok + pos
        clean_acts['embed'] = h_0

        layer = clean_model.transformer.layers[0]
        h_norm1 = layer.norm1(h_0)
        attn_out, _ = layer.self_attn(h_norm1, h_norm1, h_norm1)
        clean_acts['attn_out'] = attn_out

        h_1 = h_0 + layer.dropout1(attn_out)
        h_norm2 = layer.norm2(h_1)
        mlp_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(h_norm2))))
        clean_acts['mlp_out'] = mlp_out

    # Run corrupt model, patching the chosen activation
    with torch.no_grad():
        tok = corrupt_model.token_embed(clean_input)
        positions = torch.arange(2, device=clean_input.device).unsqueeze(0).expand(clean_input.shape[0], -1)
        pos = corrupt_model.pos_embed(positions)
        h_0 = tok + pos

        if patch_layer == 'embed':
            h_0 = clean_acts['embed']

        layer = corrupt_model.transformer.layers[0]
        h_norm1 = layer.norm1(h_0)
        attn_out, _ = layer.self_attn(h_norm1, h_norm1, h_norm1)

        if patch_layer == 'attn_out':
            attn_out = clean_acts['attn_out']

        h_1 = h_0 + layer.dropout1(attn_out)
        h_norm2 = layer.norm2(h_1)
        mlp_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(h_norm2))))

        if patch_layer == 'mlp_out':
            mlp_out = clean_acts['mlp_out']

        h_2 = h_1 + layer.dropout2(mlp_out)
        h_final = corrupt_model.ln(h_2)
        h_final_mean = h_final.mean(dim=1)
        logits = corrupt_model.output_head(h_final_mean)

    return logits


def integrated_gradients_attention(model: nn.Module, inputs: torch.Tensor, target_idx: int, steps: int = 50) -> torch.Tensor:
    """
    Compute Integrated Gradients attribution for the MultiheadAttention parameters (in_proj_weight/bias).
    Approximates the importance of the attention heads for predicting the target class.
    Returns attribution scores for each parameter.
    """
    model.eval()

    # We will compute IG with respect to the input embeddings to the attention layer.
    # To do this, we need to extract the embeddings first.

    # Get token embeddings
    tok = model.token_embed(inputs) # (batch, 2, d_model)
    positions = torch.arange(2, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
    pos = model.pos_embed(positions) # (batch, 2, d_model)

    h_0_grad = tok + pos # Initial residual stream
    h_0 = h_0_grad.detach() # Detach to prevent multiple backward passes through embeddings

    encoder_layer = model.transformer.layers[0]
    h_norm1 = encoder_layer.norm1(h_0).detach() # Baseline input to attention

    baseline = torch.zeros_like(h_norm1)
    attributions = torch.zeros_like(h_norm1)

    for alpha in torch.linspace(0, 1, steps):
        # Interpolated input
        x_alpha = baseline + alpha * (h_norm1 - baseline)
        x_alpha.requires_grad_(True)

        # Forward pass from attention onwards
        attn_out, _ = encoder_layer.self_attn(x_alpha, x_alpha, x_alpha)

        h_1 = h_0 + encoder_layer.dropout1(attn_out)
        h_norm2 = encoder_layer.norm2(h_1)
        mlp_out = encoder_layer.linear2(encoder_layer.dropout(encoder_layer.activation(encoder_layer.linear1(h_norm2))))
        h_2 = h_1 + encoder_layer.dropout2(mlp_out)

        h_final = model.ln(h_2)
        h_final_mean = h_final.mean(dim=1)
        logits = model.output_head(h_final_mean)

        # Target logit
        target_logit = logits[0, target_idx]

        # Gradients
        grad = torch.autograd.grad(target_logit, x_alpha)[0]

        attributions += grad / steps

    ig = (h_norm1 - baseline) * attributions
    # Sum over sequence length to get importance per dimension
    return ig.sum(dim=1).squeeze(0) # Shape: (d_model,)