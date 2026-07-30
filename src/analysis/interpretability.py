import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

def get_attention_head_attributions(model: nn.Module, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    Logit attribution - decompose final output into contributions from each attention head
    and each MLP neuron using direct path patching (approximation via gradient-based saliency).
    """
    model.eval()
    attributions = {}

    activations = {}

    def forward_hook(name):
        def hook(module, input, output):
            # Save activations
            if isinstance(output, tuple):
                activations[name] = output[0]
            else:
                activations[name] = output
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(forward_hook(name)))

    # Forward pass
    outputs = model(inputs)

    if targets is not None:
        batch_indices = torch.arange(outputs.size(0))
        target_logits = outputs[batch_indices, targets]
        loss = target_logits.sum()
    else:
        max_logits, _ = outputs.max(dim=-1)
        loss = max_logits.sum()

    loss.backward()

    # Calculate attribution as activation * gradient of the output wrt that activation
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight') and module.weight.grad is not None:
                # We want attribution to neurons
                # For a Linear layer, activation * weight_grad is not exactly activation grad,
                # but we can get the activation's attribution by looking at how its output
                # (which is act @ W.T + b) affects the loss.
                # Since we didn't hook tensor gradients, we will approximate using
                # the sum of absolute weight gradients connected to each input neuron.
                # Better approach: We actually just use the weight grad as importance
                attributions[name] = module.weight.grad.abs().mean(dim=1).detach() # Per-neuron attribution
        elif isinstance(module, nn.MultiheadAttention):
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight.grad is not None:
                attributions[name] = module.in_proj_weight.grad.abs().mean(dim=1).detach() # Per-head/embed attribution

    for h in hooks:
        h.remove()

    return attributions

def activation_patching(
    pure_model: nn.Module,
    collapsed_model: nn.Module,
    layer_name: str,
    inputs: torch.Tensor
) -> torch.Tensor:
    """
    Swap activations between pure (grokking) and collapsed (non-grokking) models
    at specific layers to identify WHERE grokking fails.
    """
    pure_model.eval()
    collapsed_model.eval()

    pure_activations = {}

    def pure_forward_hook(name):
        def hook(module, input, output):
            pure_activations[name] = output.detach() if not isinstance(output, tuple) else output[0].detach()
        return hook

    def collapsed_forward_patch_hook(name):
        def hook(module, input, output):
            if name in pure_activations:
                # Patch with pure model's activation
                if isinstance(output, tuple):
                    # Tuple like (output, weights) from MHA
                    return (pure_activations[name], output[1])
                return pure_activations[name]
            return output
        return hook

    # Hook pure model to save activations
    pure_hooks = []
    for name, module in pure_model.named_modules():
        if name == layer_name:
            pure_hooks.append(module.register_forward_hook(pure_forward_hook(name)))

    # Run pure model
    with torch.no_grad():
        _ = pure_model(inputs)

    # Hook collapsed model to load activations
    patch_hooks = []
    for name, module in collapsed_model.named_modules():
        if name == layer_name:
            patch_hooks.append(module.register_forward_hook(collapsed_forward_patch_hook(name)))

    # Run collapsed model with patched activations
    with torch.no_grad():
        patched_outputs = collapsed_model(inputs)

    # Cleanup
    for h in pure_hooks + patch_hooks:
        h.remove()

    return patched_outputs

def get_head_saliency_maps(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Gradient-based per-head importance scoring (saliency maps for heads).
    Requires a model with accessible attention weights.
    For ModularArithmeticTransformer, we approximate by hooking the attention outputs
    and computing their gradient with respect to the loss.
    """
    model.eval()

    n_heads = getattr(model, 'n_heads', 4)
    saliency = torch.zeros(n_heads)

    # We will hook into the MultiheadAttention layer's forward pass to capture the output tensor
    # and retain its gradient.
    attn_outputs = []

    def mha_forward_hook(module, input, output):
        # output of nn.MultiheadAttention is (attn_output, attn_output_weights)
        attn_out = output[0]
        # We need to retain grad on the tensor output
        attn_out.retain_grad()
        attn_outputs.append(attn_out)

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(mha_forward_hook))

    outputs = model(inputs)

    batch_indices = torch.arange(outputs.size(0))
    target_logits = outputs[batch_indices, targets]
    loss = target_logits.sum()

    model.zero_grad()
    loss.backward()


    # Calculate saliency per head based on gradients of attention outputs
    saliencies = []
    if len(attn_outputs) > 0:
        for attn_out in attn_outputs:
            if attn_out.grad is not None:
                grad = attn_out.grad

                embed_dim = grad.size(-1)
                head_dim = embed_dim // n_heads

                grad_heads = grad.view(*grad.shape[:-1], n_heads, head_dim)

                layer_saliency = torch.norm(grad_heads, p=2, dim=(0, 1, 3))
                if layer_saliency.sum() > 0:
                    layer_saliency = layer_saliency / layer_saliency.sum()
                saliencies.append(layer_saliency)

    if len(saliencies) > 0:
        saliency = torch.stack(saliencies).mean(dim=0)


    for h in hooks:
        h.remove()

    return saliency

def track_fourier_coefficients(model: nn.Module) -> torch.Tensor:
    """
    Fourier coefficient evolution tracking in the embedding space.
    Grokking is known to involve Fourier feature circuits.
    """
    model.eval()

    embed_weight = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and 'token' in name:
            embed_weight = module.weight.detach()
            break

    if embed_weight is None:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                embed_weight = module.weight.detach()
                break

    if embed_weight is None:
        raise ValueError("Could not find embedding layer")

    fft_coeffs = torch.fft.fft(embed_weight, dim=0).abs()

    return fft_coeffs.mean(dim=1)
