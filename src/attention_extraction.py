import torch
import torch.nn as nn
import contextlib

@contextlib.contextmanager
def extract_attention(model: nn.Module):
    """
    Context manager to extract attention weights from a PyTorch model.
    Temporarily wraps the forward pass of nn.MultiheadAttention modules
    to capture the attention weights without average_attn_weights.
    Forces eager fallback for SDPA via torch.backends.cuda.sdp_kernel(enable_math=True, ...).
    """
    attentions = []
    original_forwards = {}

    def get_forward_wrapper(module, orig_forward):
        def forward_wrapper(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            out, attn = orig_forward(*args, **kwargs)
            attentions.append(attn)
            return out, attn
        return forward_wrapper

    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                original_forwards[name] = module.forward
                module.forward = get_forward_wrapper(module, module.forward)

        # Force eager fallback for SDPA
        if hasattr(torch.backends.cuda, 'sdp_kernel'):
            with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                yield attentions
        else:
            yield attentions
    finally:
        for name, module in model.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]
