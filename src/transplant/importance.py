"""
Component importance scoring for modular arithmetic transformers.

This module computes component-level attribution via two methods:
1. Activation Patching (clean vs corrupted run metric delta)
2. Linear Proxy (weight and activation norms)

It provides a ranked table of importance for attention heads and MLP blocks.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass

try:
    from ..model import ModularArithmeticTransformer
    from ..data import generate_modular_arithmetic
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import ModularArithmeticTransformer
    from data import generate_modular_arithmetic

@dataclass
class ComponentImportance:
    """Importance score for a specific component."""
    component_name: str
    component_type: str  # 'head' or 'mlp'
    layer_idx: int
    head_idx: int = -1  # -1 for MLP
    attribution_score: float = 0.0
    weight_norm: float = 0.0
    activation_norm: float = 0.0


def compute_weight_norms(model: ModularArithmeticTransformer) -> Dict[str, float]:
    """
    Computes weight norms for all attention heads and MLP blocks.

    Args:
        model: The transformer model.

    Returns:
        A dictionary mapping component names to their weight L2 norms.
    """
    norms = {}

    for i, layer in enumerate(model.transformer.layers):
        # Attention heads
        # in_proj_weight is shape (3*d_model, d_model)
        # out_proj.weight is shape (d_model, d_model)
        d_model = model.d_model
        n_heads = model.n_heads
        d_head = d_model // n_heads

        in_proj = layer.self_attn.in_proj_weight.detach()
        out_proj = layer.self_attn.out_proj.weight.detach()

        for h in range(n_heads):
            # Q, K, V for this head
            q_norm = in_proj[h*d_head:(h+1)*d_head, :].norm().item()
            k_norm = in_proj[d_model + h*d_head:d_model + (h+1)*d_head, :].norm().item()
            v_norm = in_proj[2*d_model + h*d_head:2*d_model + (h+1)*d_head, :].norm().item()
            o_norm = out_proj[:, h*d_head:(h+1)*d_head].norm().item()

            # Combine norms as proxy (e.g. sum of squares)
            head_norm = (q_norm**2 + k_norm**2 + v_norm**2 + o_norm**2)**0.5
            norms[f"L{i}_H{h}"] = head_norm

        # MLP block (linear1 and linear2)
        l1_norm = layer.linear1.weight.detach().norm().item()
        l2_norm = layer.linear2.weight.detach().norm().item()
        norms[f"L{i}_MLP"] = (l1_norm**2 + l2_norm**2)**0.5

    return norms


def patch_activations(
    model: ModularArithmeticTransformer,
    clean_batch: Tuple[torch.Tensor, torch.Tensor],
    corrupted_batch: Tuple[torch.Tensor, torch.Tensor],
    metric_fn: callable,
) -> Dict[str, float]:
    """
    Performs activation patching on attention heads and MLP blocks.
    Runs the model on the clean batch, caches activations.
    Then runs on the corrupted batch, patching one component at a time from the clean cache.
    The importance is the delta in the metric (patched - corrupted).

    Args:
        model: The transformer model.
        clean_batch: Tuple of (inputs, targets) for the clean run.
        corrupted_batch: Tuple of (inputs, targets) for the corrupted run.
        metric_fn: Function taking (logits, targets) -> float score.

    Returns:
        Dictionary mapping component names to attribution scores.
    """
    model.eval()

    # We will use hooks to cache clean activations and patch corrupted ones.
    cache = {}

    # 1. Forward hooks for caching clean activations
    def cache_hook(name):
        def hook(module, inp, out):
            cache[name] = out[0].detach() if isinstance(out, tuple) else out.detach()
        return hook

    handles = []
    for i, layer in enumerate(model.transformer.layers):
        # We hook the self-attention layer to get the multi-head attention output
        # But wait, nn.MultiheadAttention returns (attn_output, attn_weights)
        # It's hard to split heads from its output directly without patching the inside.
        # Alternatively, we can use a forward_pre_hook on the out_proj.
        def mha_cache_hook(name):
            def hook(module, inp):
                # inp is a tuple of (q, k, v) or just input for out_proj
                cache[name] = inp[0].detach() # input to out_proj is (L, N, E) or (N, L, E)
            return hook

        handles.append(layer.self_attn.out_proj.register_forward_pre_hook(mha_cache_hook(f"L{i}_MHA_pre_out")))

        handles.append(layer.linear2.register_forward_hook(cache_hook(f"L{i}_MLP_pre_out")))

    # Run clean
    with torch.no_grad():
        x_clean, y_clean = clean_batch
        logits_clean = model(x_clean)
        clean_metric = metric_fn(logits_clean, y_clean)

    for h in handles:
        h.remove()

    # Run corrupted baseline
    with torch.no_grad():
        x_corr, y_corr = corrupted_batch
        logits_corr = model(x_corr)
        corr_metric = metric_fn(logits_corr, y_clean) # Note: evaluate against clean labels!

    # 2. Forward pre-hooks for patching
    attributions = {}

    d_model = model.d_model
    n_heads = model.n_heads
    d_head = d_model // n_heads

    for i, layer in enumerate(model.transformer.layers):
        # Patch each head
        for h in range(n_heads):
            def head_patch_hook(module, inp, head_idx=h, layer_idx=i):
                # inp is (x,). x shape: (seq_len, batch_size, d_model) if batch_first=False
                # Actually nn.TransformerEncoderLayer uses batch_first=True by default in our init
                x = inp[0].clone()
                clean_x = cache[f"L{layer_idx}_MHA_pre_out"]
                x[..., head_idx*d_head:(head_idx+1)*d_head] = clean_x[..., head_idx*d_head:(head_idx+1)*d_head]
                return (x,)

            handle = layer.self_attn.out_proj.register_forward_pre_hook(head_patch_hook)
            try:
                with torch.no_grad():
                    patched_logits = model(x_corr)
                    patched_metric = metric_fn(patched_logits, y_clean)
                    attributions[f"L{i}_H{h}"] = patched_metric - corr_metric
            finally:
                handle.remove()

        # Patch MLP
        def mlp_patch_hook(module, inp, out, layer_idx=i):
            clean_out = cache[f"L{layer_idx}_MLP_pre_out"]
            return clean_out

        handle = layer.linear2.register_forward_hook(mlp_patch_hook)
        try:
            with torch.no_grad():
                patched_logits = model(x_corr)
                patched_metric = metric_fn(patched_logits, y_clean)
                attributions[f"L{i}_MLP"] = patched_metric - corr_metric
        finally:
            handle.remove()

    return attributions


def score_components(
    model: ModularArithmeticTransformer,
    clean_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    corrupted_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> List[ComponentImportance]:
    """
    Computes importance scores for all components in the model.

    Args:
        model: The transformer model.
        clean_batch: Clean inputs and targets.
        corrupted_batch: Corrupted inputs and targets.

    Returns:
        A list of ComponentImportance objects, sorted by attribution score (descending).
    """
    norms = compute_weight_norms(model)

    attributions = {}
    if clean_batch is not None and corrupted_batch is not None:
        def accuracy_metric(logits, targets):
            preds = logits.argmax(dim=-1)
            return (preds == targets).float().mean().item()

        attributions = patch_activations(model, clean_batch, corrupted_batch, accuracy_metric)

    results = []
    for k, norm in norms.items():
        layer_idx = int(k.split('_')[0][1:])
        if 'H' in k:
            comp_type = 'head'
            head_idx = int(k.split('_')[1][1:])
        else:
            comp_type = 'mlp'
            head_idx = -1

        results.append(ComponentImportance(
            component_name=k,
            component_type=comp_type,
            layer_idx=layer_idx,
            head_idx=head_idx,
            attribution_score=attributions.get(k, 0.0),
            weight_norm=norm,
        ))

    # Sort by attribution score, then by weight norm
    results.sort(key=lambda x: (x.attribution_score, x.weight_norm), reverse=True)
    return results

if __name__ == "__main__":
    print("Testing importance module...")
    model = ModularArithmeticTransformer()
    x_clean = torch.randint(0, 59, (4, 2))
    y_clean = (x_clean[:, 0] + x_clean[:, 1]) % 59
    x_corr = torch.randint(0, 59, (4, 2))
    y_corr = (x_corr[:, 0] + x_corr[:, 1]) % 59

    scores = score_components(model, (x_clean, y_clean), (x_corr, y_corr))
    for s in scores:
        print(f"{s.component_name}: Attr={s.attribution_score:.4f}, Norm={s.weight_norm:.4f}")
