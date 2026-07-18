import torch
import torch.nn as nn
from typing import Dict, List, Optional

def compute_attention_patterns(model: nn.Module, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Extract attention patterns to identify circuits.
    Returns attention maps for all layers/heads.
    """
    patterns = {}
    hooks = []

    def get_attention(name):
        def hook(model, input, output):
            # output of MultiheadAttention is (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                # Store attention weights: (batch, num_heads, seq_len, seq_len)
                # Note: PyTorch's native MHA might return different shapes or only average attention
                # We save whatever it returns, usually (batch, seq, seq) if average
                patterns[name] = output[1].detach().cpu()
        return hook

    # Register hooks on attention layers
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            # MHA needs need_weights=True, which is default for PyTorch
            hooks.append(module.register_forward_hook(get_attention(name)))

    model.eval()
    with torch.no_grad():
        model(inputs)

    for hook in hooks:
        hook.remove()

    return patterns

def causal_patching(model: nn.Module, clean_input: torch.Tensor, corrupted_input: torch.Tensor,
                   target_layer_idx: int, metric_fn) -> float:
    """
    Perform causal patching (activation swapping) at a specific layer.
    """
    # 1. Get corrupted activations
    corrupted_activations = {}

    def get_corrupted(name):
        def hook(module, input, output):
            corrupted_activations[name] = output.detach()
        return hook

    hooks = []
    for i, layer in enumerate(model.transformer.layers):
        if i == target_layer_idx:
            hooks.append(layer.register_forward_hook(get_corrupted(f"layer_{i}")))

    with torch.no_grad():
        model(corrupted_input)

    for hook in hooks:
        hook.remove()

    # 2. Patch into clean run
    def patch_activation(name):
        def hook(module, input, output):
            # Replace with corrupted activation
            return corrupted_activations[name]
        return hook

    hooks = []
    for i, layer in enumerate(model.transformer.layers):
        if i == target_layer_idx:
            hooks.append(layer.register_forward_hook(patch_activation(f"layer_{i}")))

    with torch.no_grad():
        patched_output = model(clean_input)

    for hook in hooks:
        hook.remove()

    # 3. Compute metric
    return metric_fn(patched_output).item()

def compute_gradient_approximation(current_state: Dict[str, torch.Tensor],
                                  previous_state: Dict[str, torch.Tensor],
                                  lr: float = 1e-3,
                                  weight_decay: float = 1.0) -> Dict[str, float]:
    """
    Approximate gradient norm using consecutive weight updates: W_t - W_{t-1}.
    Assuming AdamW or SGD: delta_W approx -lr * grad - lr * wd * W_{t-1}
    => grad approx - (W_t - W_{t-1} + lr * wd * W_{t-1}) / lr
    """
    grad_norms = {
        "embedding": 0.0,
        "attention": 0.0,
        "mlp": 0.0,
        "output_head": 0.0,
        "total": 0.0
    }

    for name in current_state.keys():
        if name in previous_state and 'weight' in name and current_state[name].dtype in (torch.float32, torch.float64):
            w_t = current_state[name].float()
            w_prev = previous_state[name].float()

            # W_t - W_{t-1}
            delta_w = w_t - w_prev

            # AdamW adjustment (simplified, ignoring momentum/variance terms)
            # W_t = W_{t-1} - lr * grad - lr * wd * W_{t-1}
            # grad approx -(delta_w + lr * wd * w_prev) / lr

            # Since true Adam uses moving averages, a simpler proxy for gradient signal
            # is just the update magnitude itself (effective gradient)
            eff_grad = -delta_w / lr

            norm_sq = torch.sum(eff_grad ** 2).item()
            grad_norms["total"] += norm_sq

            if "embed" in name:
                grad_norms["embedding"] += norm_sq
            elif "transformer" in name and ("self_attn" in name or "attention" in name or "in_proj" in name or "out_proj" in name):
                grad_norms["attention"] += norm_sq
            elif "transformer" in name and "linear" in name:
                grad_norms["mlp"] += norm_sq
            elif "output_head" in name:
                grad_norms["output_head"] += norm_sq

    for k in grad_norms:
        grad_norms[k] = grad_norms[k] ** 0.5

    return grad_norms
