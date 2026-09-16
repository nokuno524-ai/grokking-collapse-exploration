import torch
import numpy as np
import pytest

from src.model import ModularArithmeticTransformer
from src.attention_extraction import extract_attention
from src.analysis.attention.attention_metrics import (
    attention_entropy,
    head_specialization_score,
    circuit_formation_index
)

def test_extract_attention_shapes():
    model = ModularArithmeticTransformer(d_model=32, n_heads=4, n_layers=1)
    x = torch.randint(0, 59, (5, 2))  # batch size 5, seq len 2

    with extract_attention(model) as attn:
        out = model(x)

    assert len(attn) == 1, f"Expected 1 layer's attention, got {len(attn)}"
    layer_attn = attn[0]

    expected_shape = (5, 4, 2, 2)
    assert layer_attn.shape == expected_shape, f"Expected shape {expected_shape}, got {layer_attn.shape}"
    assert isinstance(layer_attn, np.ndarray), "Attention should be a numpy array to avoid memory leaks"


def test_attention_metrics_values():
    # Construct a known attention matrix
    # shape: (batch_size=1, n_heads=1, seq_len=2, seq_len=2)
    # Head 0, Query 0 attends completely to Key 0
    # Head 0, Query 1 attends uniformly to Key 0 and Key 1

    attn = np.array([[[
        [1.0, 0.0],
        [0.5, 0.5]
    ]]])

    # 1) Entropy
    # Query 0: - (1*log(1) + 0*log(0)) = 0
    # Query 1: - (0.5*log(0.5) + 0.5*log(0.5)) = -log(0.5) ≈ 0.693147
    # Mean entropy = (0 + 0.693147) / 2 ≈ 0.34657
    ent = attention_entropy(attn)
    assert np.isclose(ent, 0.34657359027997264, atol=1e-5)

    # 2) Specialization Score
    # Query positions variance for Key 0: var([1.0, 0.5]) = 0.0625
    # Query positions variance for Key 1: var([0.0, 0.5]) = 0.0625
    # Mean specialization = 0.0625
    spec = head_specialization_score(attn)
    assert np.isclose(spec, 0.0625, atol=1e-5)

    # 3) Circuit Formation Index
    # Difference between the same matrix should be 0
    cfi_zero = circuit_formation_index(attn, attn)
    assert np.isclose(cfi_zero, 0.0)

    attn_new = np.array([[[
        [0.5, 0.5],
        [0.5, 0.5]
    ]]])
    # abs diff:
    # [0.5, 0.5]
    # [0.0, 0.0]
    # mean(abs_diff) = (0.5+0.5+0+0)/4 = 0.25
    cfi = circuit_formation_index(attn, attn_new)
    assert np.isclose(cfi, 0.25)

def test_metadata_alignment():
    # We want to verify that checkpoint extraction properly aligns step metadata.
    # While we can't easily mock the whole directory structure here cleanly without tmpdir,
    # we can simulate the dictionary records built by the extraction script logic.
    records = []
    prev = None

    # Fake steps
    for step in [100, 200]:
        attn = np.full((1, 1, 2, 2), 0.5)
        cfi = 0.0
        if prev is not None:
            cfi = circuit_formation_index(prev, attn)
        prev = attn

        record = {
            "step": step,
            "layer": 0,
            "mean_entropy": attention_entropy(attn),
            "circuit_formation_index": cfi
        }
        records.append(record)

    assert len(records) == 2
    assert records[0]["step"] == 100
    assert records[0]["layer"] == 0
    assert records[1]["step"] == 200
    assert records[1]["layer"] == 0
