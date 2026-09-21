import pytest
import torch
import torch.nn as nn
from src.attention_extraction import extract_attention
from src.model import ModularArithmeticTransformer

def test_extract_attention_context_manager():
    # Small test model
    model = ModularArithmeticTransformer(
        prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=2
    )
    model.eval()

    # 2 layers, 1 multihead attention each
    x = torch.randint(0, 5, (4, 2))

    with extract_attention(model) as extracted:
        _ = model(x)

    # Check that we extracted attention patterns
    assert len(extracted) > 0, "No attention weights extracted"

    # The dictionary keys should map to module names or generic names
    for name, patterns in extracted.items():
        assert isinstance(patterns, list)
        assert len(patterns) == 1, "Expected 1 forward pass per module"

        attn = patterns[0]
        # Shape should be (batch_size, n_heads, tgt_len, src_len)
        assert attn.shape == (4, 2, 2, 2)

        # Verify it's a valid probability distribution (sums to 1 over src_len)
        row_sums = attn.sum(axis=-1)
        import numpy as np
        assert np.allclose(row_sums, 1.0, atol=1e-5), "Attention weights do not sum to 1"

def test_original_forward_restored():
    model = ModularArithmeticTransformer(
        prime=5, d_model=16, n_heads=2, d_ff=32, n_layers=1
    )
    model.eval()

    # Capture original forward
    original_forwards = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            original_forwards[name] = module.forward

    x = torch.randint(0, 5, (4, 2))

    with extract_attention(model) as extracted:
        _ = model(x)

    # Check that original forwards are restored
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            assert module.forward == original_forwards[name], "Original forward not restored"
