import pytest
import torch
import numpy as np
from typing import List, Dict
import copy

from src.attention.extraction import compute_entropy, extract_attention, compute_head_features
from src.attention.taxonomy import cluster_heads
from src.attention.causal import ablate_head, evaluate_ablation
from src.model import ModularArithmeticTransformer

def test_extraction_synthetic():
    B, H, S = 2, 4, 3
    # Perfect uniform attention
    attn_weights = np.ones((B, H, S, S)) / S

    # We will simulate masking
    mask = np.ones((B, S), dtype=bool)
    mask[1, 2] = False # Sequence 2 has padding at pos 2

    tokens = np.array([
        [10, 20, 10], # seq 1
        [15, 15, 0],  # seq 2
    ])

    features = compute_head_features(attn_weights, tokens, mask)

    # Check entropy
    # For seq 1, S=3, entropy is log(3) ~= 1.098
    # For seq 2, S=3 but mask has 2 valid, though attn_weights has 1/3 for all 3.
    # Our function renormalizes if we look at compute_entropy tensor level, but wait,
    # compute_head_features converts mask to tensor and calls compute_entropy.
    # Actually for perfect uniform, without proper k_masking in the numpy input,
    # it gets masked in compute_entropy.

    assert "entropy" in features
    assert "diagonal_mass" in features
    assert "offset_1_mass" in features
    assert "offset_2_mass" in features
    assert "copy_score" in features
    assert "induction_score" in features


def test_taxonomy():
    # Synthetic features
    # Head 0: Positional (offset_1)
    # Head 1: Copy
    # Head 2: Induction
    # Head 3: Other (uniform)

    B, H = 1, 4
    features = {
        'offset_1_mass': np.array([[0.8, 0.1, 0.1, 0.1]]),
        'offset_2_mass': np.array([[0.1, 0.1, 0.1, 0.1]]),
        'copy_score': np.array([[0.1, 0.9, 0.1, 0.1]]),
        'induction_score': np.array([[0.1, 0.1, 0.85, 0.1]]),
    }

    classes = cluster_heads(features, threshold=0.3)

    assert classes[0] == "positional"
    assert classes[1] == "copy"
    assert classes[2] == "induction"
    assert classes[3] == "other"

def test_causal_ablation():
    model = ModularArithmeticTransformer(d_model=16, n_heads=4, d_ff=32, n_layers=1)
    # Give it some random weights
    for p in model.parameters():
        p.data.normal_(0, 1)

    mha = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            mha = module
            break

    assert mha is not None

    # Store old weight
    old_in_proj_weight = mha.in_proj_weight.clone()

    # Ablate head 1
    ablate_head(model, head_idx=1, method="zero")

    # Check that head 1 is zeroed out in Q, K, V
    d_model = 16
    d_head = 4
    start_idx = 4
    end_idx = 8

    for offset in [0, d_model, 2 * d_model]:
        # Head 1 is zero
        assert torch.all(mha.in_proj_weight[offset + start_idx : offset + end_idx, :] == 0)
        # Head 0 is NOT zero
        assert not torch.all(mha.in_proj_weight[offset + 0 : offset + 4, :] == 0)

    # Check that out_proj for head 1 is zero
    assert torch.all(mha.out_proj.weight[:, start_idx:end_idx] == 0)
    assert not torch.all(mha.out_proj.weight[:, 0:4] == 0)

    # Test mean ablation
    ablate_head(model, head_idx=2, method="mean")
    for offset in [0, d_model, 2 * d_model]:
        # Head 2 is mean
        block = mha.in_proj_weight[offset + 8 : offset + 12, :]
        assert torch.allclose(block, block.mean() * torch.ones_like(block))

def test_extract_attention_hook():
    model = ModularArithmeticTransformer(d_model=16, n_heads=4, d_ff=32, n_layers=1)

    x = torch.randint(0, 59, (2, 2))

    with extract_attention(model) as attentions:
        out = model(x)

    assert len(attentions) == 1
    attn_weights = attentions[0]
    assert attn_weights.shape == (2, 4, 2, 2)

    # Check that sums are 1
    assert np.allclose(attn_weights.sum(axis=-1), 1.0)

def test_compute_entropy_no_nans():
    B, H, S = 2, 1, 3
    # Make one element exactly 0.0 to test clamp logic
    attn_weights = torch.zeros(B, H, S, S)
    attn_weights[:, :, :, 0] = 1.0 # Everything attends to token 0

    # We should not get NaNs
    entropy = compute_entropy(attn_weights)
    assert not torch.isnan(entropy).any()

    # Test with mask
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, 2] = False # token 2 is padding for batch 1

    entropy_masked = compute_entropy(attn_weights, mask)
    assert not torch.isnan(entropy_masked).any()
