import torch
import torch.nn as nn
from src.composition import analyze_composition, composition_matrix, detect_circuits

class MockAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Manually set weights for predictability
        self.in_proj_weight = nn.Parameter(torch.eye(d_model * 3, d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj.weight.data = torch.eye(d_model, d_model)

class MockLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn = MockAttention(d_model, n_heads)

class MockTransformer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(d_model, n_heads)])

class MockModel(nn.Module):
    def __init__(self, d_model=4, n_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.transformer = MockTransformer(d_model, n_heads)

def test_composition():
    # 2-head mock model
    model = MockModel(d_model=4, n_heads=2)
    inputs = torch.zeros(1, 2)

    # Analysis
    scores = analyze_composition(model, inputs)

    # Because weights are Identity, orthogonal parts of space mean
    # cross-head composition should be 0, and self-head composition should be 1.0 (if normalized properly)
    q_comp = scores['q_composition']
    assert q_comp.shape == (2, 2)

    # Trace/diagonal should be higher than off-diagonal for Identity weights
    assert q_comp[0, 0].item() > q_comp[0, 1].item()
    assert q_comp[1, 1].item() > q_comp[1, 0].item()

    # Full matrix
    mat = composition_matrix(model, None)
    assert mat.shape == (2, 2)

    # Detect circuits
    # Since self-composition is high, we should detect (0,0) and (1,1)
    # Threshold might need tuning but identity yields ~1.0 on diagonal
    circuits = detect_circuits(mat, threshold=0.2)
    assert (0, 0) in circuits
    assert (1, 1) in circuits
    assert (0, 1) not in circuits
