import torch
import pytest
import numpy as np
from pathlib import Path
from src.model import ModularArithmeticTransformer
from src.weight_analysis import (
    track_weight_norm_distribution,
    get_svd_spectrum
)

@pytest.fixture
def mock_model():
    return ModularArithmeticTransformer(
        prime=5,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1
    )

def test_track_weight_norm_distribution(mock_model):
    norms = track_weight_norm_distribution(mock_model)

    assert isinstance(norms, dict)
    assert "token_embed.weight" in norms
    assert "output_head.weight" in norms
    assert "transformer.layers.0.self_attn.in_proj_weight" in norms

    for k, v in norms.items():
        assert isinstance(v, float)
        assert v >= 0.0

def test_get_svd_spectrum(mock_model):
    s_embed = get_svd_spectrum(mock_model, "token_embed.weight")

    assert isinstance(s_embed, np.ndarray)
    assert len(s_embed) > 0

    # Singular values should be sorted descending
    assert all(s_embed[i] >= s_embed[i+1] - 1e-6 for i in range(len(s_embed)-1))

    # Test on a linear layer
    s_out = get_svd_spectrum(mock_model, "output_head.weight")
    assert isinstance(s_out, np.ndarray)
    assert len(s_out) > 0

    # Test error handling
    with pytest.raises(ValueError):
        get_svd_spectrum(mock_model, "non_existent_layer")
