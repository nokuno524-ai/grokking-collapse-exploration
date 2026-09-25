import torch
import torch.nn as nn

mha = nn.MultiheadAttention(16, 2)
original_forward = mha.forward

def patched_forward(self, *args, **kwargs):
    # args comes without self if we do it via __get__? No, if we do __get__, self IS the first argument!

    # Let's inspect the signature of original_forward
    import inspect
    sig = inspect.signature(original_forward)

    # sig for original_forward does NOT include self, because original_forward is already a bound method!
    # Wait, if we use __get__, it binds our function to the instance, so our function receives self as the first arg.
    # But original_forward is ALREADY bound, so we don't pass self to it.

    # We must remove self before passing to original_forward.
    # And we must remove self before binding arguments to the sig.

    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    bound.arguments['need_weights'] = True
    bound.arguments['average_attn_weights'] = False

    # Remove key_padding_mask if multiple values issue occurs? The issue is sig.bind doesn't have self.

    return original_forward(*bound.args, **bound.kwargs)

mha.forward = patched_forward.__get__(mha, type(mha))
q = torch.randn(4, 2, 16)
try:
    mha(q, q, q, key_padding_mask=None, need_weights=False, average_attn_weights=True)
except Exception as e:
    print(e)
