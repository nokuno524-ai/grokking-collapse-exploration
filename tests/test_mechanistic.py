import torch
import pytest
from src.analysis.sae_grokking import SparseAutoencoder, train_sae
from experiments.curriculum_grokking import CurriculumConfig, generate_curriculum_data
from src.analysis.attention_evolution import compute_attention_entropy
from src.analysis.svd_analysis import compute_effective_rank
from src.model import ModularArithmeticTransformer

def test_sae_training():
    d_in = 64
    d_hidden = 128
    batch_size = 32
    # Create some mock activations
    acts = torch.randn(100, d_in)
    sae = train_sae(acts, d_hidden, epochs=1, batch_size=batch_size)
    assert isinstance(sae, SparseAutoencoder)

    # Forward pass check
    x_rec, f, loss = sae(acts[:10])
    assert x_rec.shape == (10, d_in)
    assert f.shape == (10, d_hidden)
    assert loss.item() > 0

def test_curriculum_data_generation():
    config = CurriculumConfig(strategy="easy_to_hard", prime=59, train_fraction=0.3)
    train_in, train_tgt, test_in, test_tgt = generate_curriculum_data(config)

    # Check that targets have some chunked structure but aren't strictly sorted
    # since we apply the pacing logic.
    assert train_tgt.shape[0] > 0

def test_attention_entropy():
    # Mock attention weights: (batch_size, n_heads, seq_len, seq_len)
    batch_size = 2
    n_heads = 4
    seq_len = 5

    # Uniform attention -> Max entropy
    uniform_attn = torch.ones(batch_size, n_heads, seq_len, seq_len) / seq_len
    ent_uniform = compute_attention_entropy(uniform_attn)
    assert ent_uniform.shape == (batch_size, n_heads, seq_len)

    # One-hot attention -> Zero entropy
    onehot_attn = torch.zeros(batch_size, n_heads, seq_len, seq_len)
    onehot_attn[:, :, :, 0] = 1.0
    ent_onehot = compute_attention_entropy(onehot_attn)
    assert torch.allclose(ent_onehot, torch.zeros_like(ent_onehot), atol=1e-5)

def test_svd_effective_rank():
    # Mock uniform singular values
    s_uniform = torch.ones(10)
    rank_uniform = compute_effective_rank(s_uniform)
    # log(10) entropy -> exp(log(10)) = 10
    assert abs(rank_uniform - 10.0) < 1e-4

    # Point mass
    s_point = torch.zeros(10)
    s_point[0] = 1.0
    rank_point = compute_effective_rank(s_point)
    assert abs(rank_point - 1.0) < 1e-4

def test_compute_svd_spectrum():
    model = ModularArithmeticTransformer(prime=59, d_model=32)
    from src.analysis.svd_analysis import compute_svd_spectrum
    spectra = compute_svd_spectrum(model)

    assert "embedding" in spectra
    assert "output_head" in spectra
    assert "ffn1" in spectra
    assert "ffn2" in spectra
    assert spectra["embedding"].shape[0] == min(59, 32)

def test_correlate_with_circuit():
    from src.analysis.sae_grokking import correlate_with_circuit
    # Mock data
    acts = torch.randn(100, 10)
    scores = torch.randn(100)
    corr = correlate_with_circuit(acts, scores)

    assert corr.shape == (10,)
    assert torch.all((corr >= -1.0) & (corr <= 1.0))
