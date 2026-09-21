import pytest
import numpy as np
from src.analysis.attention_metrics import (
    compute_attention_entropy,
    compute_diagonal_mass,
    compute_head_specialization
)
from src.analysis.atlas_analysis import render_atlas_figure
import base64

def test_compute_attention_entropy():
    # Uniform attention over 4 tokens
    # log(4) approx 1.386
    attn = np.ones((2, 2, 4, 4)) / 4
    entropy = compute_attention_entropy(attn)
    assert entropy.shape == (2, 2, 4)
    assert np.allclose(entropy, np.log(4))

    # One-hot attention (0 entropy)
    attn_onehot = np.eye(4)[None, None, :, :]
    entropy = compute_attention_entropy(attn_onehot)
    assert np.allclose(entropy, 0.0, atol=1e-5)

def test_compute_diagonal_mass():
    # Identity matrix means 100% on diagonal
    attn = np.eye(4)[None, None, :, :]
    diag_mass = compute_diagonal_mass(attn)
    assert diag_mass.shape == (1, 1)
    assert np.allclose(diag_mass, 1.0)

    # Uniform matrix means 1/4 on diagonal
    attn_uniform = np.ones((2, 2, 4, 4)) / 4
    diag_mass = compute_diagonal_mass(attn_uniform)
    assert np.allclose(diag_mass, 0.25)

def test_compute_head_specialization():
    # Two identical heads -> 0 variance
    attn_identical = np.ones((1, 2, 4, 4)) / 4
    spec = compute_head_specialization(attn_identical)
    assert np.isclose(spec, 0.0)

    # Two totally different heads
    head1 = np.eye(4)
    head2 = np.ones((4, 4)) / 4
    attn_diff = np.stack([head1, head2])[None, ...] # shape (1, 2, 4, 4)
    spec = compute_head_specialization(attn_diff)
    assert spec > 0.0

def test_render_atlas_figure():
    attn = np.random.rand(4, 2, 2)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    b64_img = render_atlas_figure(attn, "Test Title")

    # Check it returned a non-empty string that decodes properly
    assert isinstance(b64_img, str)
    assert len(b64_img) > 100

    # Should be valid base64
    decoded = base64.b64decode(b64_img)
    # Check PNG magic bytes
    assert decoded[:8] == b'\x89PNG\r\n\x1a\n'
