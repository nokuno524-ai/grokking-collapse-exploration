import torch
import pytest
from src.attention_taxonomy import (
    detect_previous_token_heads,
    detect_induction_heads,
    detect_duplicate_token_heads
)

def test_detect_previous_token_heads():
    # shape: (batch, n_heads, seq_len, seq_len)
    # 1 batch, 2 heads, seq_len=4
    attn_weights = torch.zeros(1, 2, 4, 4)

    # Head 0: attends purely to previous token (i-1)
    for i in range(1, 4):
        attn_weights[0, 0, i, i-1] = 1.0

    # Head 1: attends purely to current token (i)
    for i in range(4):
        attn_weights[0, 1, i, i] = 1.0

    prev_heads = detect_previous_token_heads(attn_weights, threshold=0.9)
    assert 0 in prev_heads
    assert 1 not in prev_heads

def test_detect_induction_heads():
    # Sequence: [A, B, C, A, B]
    # Indices:   0  1  2  3  4
    # Duplicate prefix is at index 0 (A) and index 3 (A)
    # The token following the prefix is at index 1 (B)
    # Therefore, at index 4 (B), an induction head should attend to index 1 (B)

    inputs = torch.tensor([[10, 20, 30, 10, 20]])
    attn_weights = torch.zeros(1, 2, 5, 5)

    # Head 0: Induction head pattern
    # At pos 4, attends to pos 1
    attn_weights[0, 0, 4, 1] = 1.0

    # Head 1: Random attention
    attn_weights[0, 1, 4, 0] = 1.0

    ind_heads = detect_induction_heads(attn_weights, inputs, threshold=0.9)
    assert 0 in ind_heads
    assert 1 not in ind_heads

def test_detect_duplicate_token_heads():
    # Sequence: [A, B, C, A, D]
    # Indices:   0  1  2  3  4
    # Duplicate is at index 3 (A) which matches index 0 (A)

    inputs = torch.tensor([[10, 20, 30, 10, 40]])
    attn_weights = torch.zeros(1, 2, 5, 5)

    # Head 0: Duplicate token head
    # At pos 3, attends to pos 0
    attn_weights[0, 0, 3, 0] = 1.0

    # Head 1: Random attention
    attn_weights[0, 1, 3, 2] = 1.0

    dup_heads = detect_duplicate_token_heads(attn_weights, inputs, threshold=0.9)
    assert 0 in dup_heads
    assert 1 not in dup_heads
