"""
Circuit discovery tools for modular arithmetic transformer.
Implements activation patching, path patching, and logit attribution.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

def activation_patching(
    model_receiver: nn.Module,
    model_donor: nn.Module,
    component_name: str,
    data: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Patch activations from model_donor into model_receiver at a specific component.

    Args:
        model_receiver: The model to patch into (e.g. collapsed model)
        model_donor: The model to get activations from (e.g. grokked model)
        component_name: Name of the component to patch ('token_embed', 'pos_embed', 'transformer')
        data: Input tensor of shape (batch, 2)

    Returns:
        Tuple of (original_logits, patched_logits)
    """
    device = data.device
    batch_size = data.shape[0]

    # Run donor model to get its intermediate activations
    donor_activations = {}

    def donor_hook(module, input, output, name):
        donor_activations[name] = output.detach()

    # Register hooks on donor
    donor_hooks = []
    if component_name == 'token_embed' and hasattr(model_donor, 'token_embed'):
        donor_hooks.append(model_donor.token_embed.register_forward_hook(
            lambda m, i, o: donor_hook(m, i, o, 'token_embed')))
    elif component_name == 'pos_embed' and hasattr(model_donor, 'pos_embed'):
        donor_hooks.append(model_donor.pos_embed.register_forward_hook(
            lambda m, i, o: donor_hook(m, i, o, 'pos_embed')))
    elif component_name == 'transformer' and hasattr(model_donor, 'transformer'):
        donor_hooks.append(model_donor.transformer.register_forward_hook(
            lambda m, i, o: donor_hook(m, i, o, 'transformer')))

    # Forward donor to populate activations
    with torch.no_grad():
        _ = model_donor(data)

    for h in donor_hooks:
        h.remove()

    # Run receiver model normally
    with torch.no_grad():
        orig_logits = model_receiver(data)

    # Run receiver model with patch
    def patch_hook(module, input, output, name):
        if name in donor_activations:
            return donor_activations[name]
        return output

    receiver_hooks = []
    if component_name == 'token_embed' and hasattr(model_receiver, 'token_embed'):
        receiver_hooks.append(model_receiver.token_embed.register_forward_hook(
            lambda m, i, o: patch_hook(m, i, o, 'token_embed')))
    elif component_name == 'pos_embed' and hasattr(model_receiver, 'pos_embed'):
        receiver_hooks.append(model_receiver.pos_embed.register_forward_hook(
            lambda m, i, o: patch_hook(m, i, o, 'pos_embed')))
    elif component_name == 'transformer' and hasattr(model_receiver, 'transformer'):
        receiver_hooks.append(model_receiver.transformer.register_forward_hook(
            lambda m, i, o: patch_hook(m, i, o, 'transformer')))

    with torch.no_grad():
        patched_logits = model_receiver(data)

    for h in receiver_hooks:
        h.remove()

    return orig_logits, patched_logits


def path_patching(model: nn.Module, head_idx: int, data: torch.Tensor) -> torch.Tensor:
    """
    Trace information flow through a specific attention head.
    Since we use nn.TransformerEncoderLayer, we have to extract the Q, K, V weights
    and manually run the head to get its specific output path.

    Args:
        model: The ModularArithmeticTransformer
        head_idx: The index of the head to patch
        data: Input tensor

    Returns:
        The output logits if only this head contributed to the output
    """
    device = data.device
    batch_size = data.shape[0]

    with torch.no_grad():
        tok = model.token_embed(data)
        positions = torch.arange(2, device=device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        # In a 1-layer transformer, we can extract the first layer's self-attn
        layer = model.transformer.layers[0]
        self_attn = layer.self_attn

        # Dimensions
        embed_dim = self_attn.embed_dim
        num_heads = self_attn.num_heads
        head_dim = embed_dim // num_heads

        # Get in_proj weights (they are concatenated Q, K, V)
        in_proj_weight = self_attn.in_proj_weight
        in_proj_bias = self_attn.in_proj_bias

        # Q, K, V are each embed_dim x embed_dim
        q_w, k_w, v_w = in_proj_weight.chunk(3, dim=0)
        q_b, k_b, v_b = in_proj_bias.chunk(3, dim=0)

        # Project inputs for this specific head
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim

        head_q_w = q_w[start_idx:end_idx, :]
        head_k_w = k_w[start_idx:end_idx, :]
        head_v_w = v_w[start_idx:end_idx, :]

        head_q_b = q_b[start_idx:end_idx]
        head_k_b = k_b[start_idx:end_idx]
        head_v_b = v_b[start_idx:end_idx]

        q = torch.nn.functional.linear(h, head_q_w, head_q_b)
        k = torch.nn.functional.linear(h, head_k_w, head_k_b)
        v = torch.nn.functional.linear(h, head_v_w, head_v_b)

        # Attention scores for this head
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (head_dim ** 0.5)
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Head output
        head_out = torch.bmm(attn_probs, v)

        # Project back out using the out_proj for this specific head
        out_proj_w = self_attn.out_proj.weight[:, start_idx:end_idx]
        # We don't add out_proj_bias here because we're isolating the head's contribution
        # and bias would be added for all heads combined. We just want the head's path.

        head_contrib = torch.nn.functional.linear(head_out, out_proj_w)

        # To get the logit contribution, we pass this through the output head
        # We bypass the FFN and LayerNorm as path patching isolates the linear path
        head_contrib_mean = head_contrib.mean(dim=1)
        logits_contrib = model.output_head(head_contrib_mean)

        return logits_contrib
def logit_attribution(model: nn.Module, data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Decompose model output by component (e.g., token embed vs pos embed).

    Args:
        model: The ModularArithmeticTransformer
        data: Input tensor

    Returns:
        Dictionary mapping component names to their logit attribution (batch, prime)
    """
    # The output logit is approximately linear with respect to the input to the output head
    # (ignoring LayerNorm non-linearities for simplicity of attribution)

    batch_size = data.shape[0]
    device = data.device

    # 1. Get embeddings
    with torch.no_grad():
        tok = model.token_embed(data)  # (batch, 2, d_model)

        positions = torch.arange(2, device=device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)  # (batch, 2, d_model)

        # 2. Approximate transformer as passing through these components
        # A true logit attribution requires linearizing the transformer and layer norm.
        # We will provide a simplified attribution: passing each directly to the output head

        tok_mean = tok.mean(dim=1)
        pos_mean = pos.mean(dim=1)

        attr_tok = model.output_head(tok_mean)
        attr_pos = model.output_head(pos_mean)

        return {
            'token_embed_direct': attr_tok,
            'pos_embed_direct': attr_pos
        }

def compare_circuits(model_a: nn.Module, model_b: nn.Module) -> Dict[str, float]:
    """
    Compare the structural differences between two models (e.g. grokked vs collapsed).

    Args:
        model_a: First model
        model_b: Second model

    Returns:
        Dict mapping component name to L2 distance of weights
    """
    differences = {}

    for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if name_a == name_b:
            diff = torch.norm(param_a.detach() - param_b.detach()).item()
            differences[name_a] = diff

    return differences
