import torch
import pytest
from src.model import ModularArithmeticTransformer
from src.analysis.feature_evolution import extract_fourier_features, extract_attention_statistics

def test_extract_fourier_features():
    model = ModularArithmeticTransformer()
    features = extract_fourier_features(model)

    assert 'peak_magnitude' in features
    assert 'entropy' in features

    # Should be valid floats
    assert isinstance(features['peak_magnitude'], float)
    assert isinstance(features['entropy'], float)
    assert features['peak_magnitude'] >= 0
    assert features['entropy'] >= 0

def test_extract_attention_statistics():
    model = ModularArithmeticTransformer()
    x = torch.randint(0, 59, (4, 2))

    stats = extract_attention_statistics(model, x)

    assert 'avg_entropy' in stats
    assert 'head_entropies' in stats

    assert isinstance(stats['avg_entropy'], float)
    assert isinstance(stats['head_entropies'], list)
    assert len(stats['head_entropies']) == model.n_heads

    # Entropies should be positive
    assert stats['avg_entropy'] >= 0
    for he in stats['head_entropies']:
        assert isinstance(he, float)
        assert he >= 0
