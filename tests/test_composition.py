import torch
import torch.nn as nn
from src.analysis.composition import compute_ov_composition, compute_qk_composition, get_head_matrices

def test_compute_ov_composition():
    # Define arbitrary matrices
    # d_model = 16, d_head = 4
    W_V = torch.randn(4, 16)
    W_O = torch.randn(16, 4)

    OV = compute_ov_composition(W_V, W_O)

    assert OV.shape == (16, 16)
    assert torch.allclose(OV, W_O @ W_V)

def test_compute_qk_composition():
    # d_model = 16, d_head = 4
    W_Q = torch.randn(4, 16)
    W_K = torch.randn(4, 16)

    QK = compute_qk_composition(W_Q, W_K)

    assert QK.shape == (16, 16)
    assert torch.allclose(QK, W_Q.t() @ W_K)

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 16
        self.n_heads = 4

        # Mocking transformer.layers[0].self_attn
        class MockAttention:
            def __init__(self):
                self._qkv_same_embed_dim = True
                self.in_proj_weight = torch.randn(3 * 16, 16)

                class OutProj:
                    def __init__(self):
                        self.weight = torch.randn(16, 16)
                self.out_proj = OutProj()

        class MockLayer:
            def __init__(self):
                self.self_attn = MockAttention()

        class MockTransformer:
            def __init__(self):
                self.layers = [MockLayer()]

        self.transformer = MockTransformer()

def test_get_head_matrices():
    model = MockModel()

    matrices = get_head_matrices(model, head_idx=1, layer_idx=0)

    assert "W_Q" in matrices
    assert "W_K" in matrices
    assert "W_V" in matrices
    assert "W_O" in matrices

    # Check shapes
    assert matrices["W_Q"].shape == (4, 16)
    assert matrices["W_K"].shape == (4, 16)
    assert matrices["W_V"].shape == (4, 16)
    assert matrices["W_O"].shape == (16, 4)
