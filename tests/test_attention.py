import torch
import torch.nn as nn
import numpy as np
import pytest
from src.attention_extraction import extract_attention

def test_extract_attention():
    # Setup tiny synthetic transformer config
    layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
    model = nn.TransformerEncoder(layer, num_layers=1)

    # Needs to be tested in train and eval modes due to fast-path intricacies
    modes = [True, False]
    x = torch.randn(2, 4, 16) # batch=2, seq=4, d_model=16

    for train_mode in modes:
        model.train(train_mode)
        with torch.no_grad():
            with extract_attention(model) as attns:
                out = model(x)

        assert len(attns) == 1
        attn = attns[0]
        # shape should be (batch_size, num_heads, seq_len, seq_len)
        assert attn.shape == (2, 2, 4, 4)

        # Verify it's a valid probability distribution
        # sum along last dimension should be ~1.0
        sums = np.sum(attn, axis=-1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5, atol=1e-5)

def test_compute_entropy():
    # Need to import compute_entropy from scripts since we implemented it there
    # It would be better structure to have compute_entropy in src, but let's test it via sys.path
    from src.attention_extraction import compute_entropy

    # 1. Perfect uniformity: max entropy
    n = 4
    uniform_attn = np.ones((1, 1, 1, n)) / n
    expected_entropy = -np.log(1/n) # ln(4)
    ent = compute_entropy(uniform_attn)
    np.testing.assert_allclose(ent, [[[expected_entropy]]], rtol=1e-5)

    # 2. Perfect delta distribution: 0 entropy
    delta_attn = np.zeros((1, 1, 1, n))
    delta_attn[0, 0, 0, 0] = 1.0
    ent = compute_entropy(delta_attn)
    np.testing.assert_allclose(ent, [[[0.0]]], rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main(["-v", "test_attention.py"])
