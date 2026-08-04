import torch
import pytest
from src.model import ModularArithmeticTransformer
from src.analysis.jlens import JLensAnalyzer

def test_jlens_extraction():
    model = ModularArithmeticTransformer()
    analyzer = JLensAnalyzer(model)
    x = torch.randint(0, 59, (4, 2))

    reps = analyzer.extract_representations(x)

    assert 'embedding' in reps
    assert 'transformer' in reps
    assert 'layer_norm' in reps

    # Check shapes
    assert reps['embedding'].shape == (4, 2, 128)
    assert reps['transformer'].shape == (4, 2, 128)
    assert reps['layer_norm'].shape == (4, 2, 128)

def test_jlens_projection():
    model = ModularArithmeticTransformer()
    analyzer = JLensAnalyzer(model)
    x = torch.randint(0, 59, (4, 2))

    reps = analyzer.extract_representations(x)
    projs = analyzer.project_to_vocabulary(reps)

    assert 'embedding' in projs
    assert 'transformer' in projs
    assert 'layer_norm' in projs

    # Check shapes (batch, prime)
    assert projs['embedding'].shape == (4, 59)
    assert projs['transformer'].shape == (4, 59)
    assert projs['layer_norm'].shape == (4, 59)

def test_jlens_dimensionality_and_metrics():
    model = ModularArithmeticTransformer()
    analyzer = JLensAnalyzer(model)
    x = torch.randint(0, 59, (4, 2))

    metrics = analyzer.analyze(x)

    for key in ['embedding', 'transformer', 'layer_norm']:
        assert key in metrics
        assert 'entropy' in metrics[key]
        assert 'rank' in metrics[key]

        # Entropy and rank should be positive floats
        assert isinstance(metrics[key]['entropy'], float)
        assert isinstance(metrics[key]['rank'], float)
        assert metrics[key]['entropy'] > 0
        assert metrics[key]['rank'] > 0
