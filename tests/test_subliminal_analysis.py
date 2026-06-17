import torch
import numpy as np
from src.model import ModularArithmeticTransformer
from analysis.subliminal_detection import (
    detect_subliminal_traits,
    measure_trait_transfer,
    trait_projection_score,
    bootstrap_ci
)


def test_detect_subliminal_traits():
    model = ModularArithmeticTransformer(prime=5, d_model=8)
    teacher_outputs = torch.randn(10, 5)
    sae_features = torch.randn(8, 3) # (d_model, n_features)

    score = detect_subliminal_traits(teacher_outputs, model, sae_features)
    assert isinstance(score, float)
    assert score >= 0.0


def test_measure_trait_transfer():
    model = ModularArithmeticTransformer(prime=5, d_model=8)
    # Mock inputs: batch size 4, seq length 2
    control_data = torch.randint(0, 5, (4, 2))
    teacher_data = torch.randint(0, 5, (4, 2))

    score = measure_trait_transfer(control_data, teacher_data, model)
    assert isinstance(score, float)
    assert score >= 0.0


def test_trait_projection_score():
    model = ModularArithmeticTransformer(prime=5, d_model=8)
    trait_direction = torch.randn(8) # (d_model,)

    score = trait_projection_score(model, trait_direction)
    assert isinstance(score, float)
    assert score >= 0.0


def test_bootstrap_ci():
    # Provide enough variance to avoid degenerate bootstrap
    data = np.array([1.0, 1.2, 0.9, 1.1, 1.05, 1.15, 0.95, 1.0])

    low, high = bootstrap_ci(data, confidence_level=0.95)
    assert isinstance(low, float)
    assert isinstance(high, float)
    assert low < high
    assert low <= np.mean(data) <= high

    # Test flat data case
    flat_data = np.array([2.0, 2.0, 2.0, 2.0])
    low_f, high_f = bootstrap_ci(flat_data)
    assert isinstance(low_f, float)
    assert isinstance(high_f, float)
    # The interval should tightly bound the mean (2.0)
    assert abs(low_f - 2.0) < 0.1
    assert abs(high_f - 2.0) < 0.1


def test_track_traits_during_collapse():
    from experiments.collapse_subliminal import track_traits_during_collapse
    model = ModularArithmeticTransformer(prime=5, d_model=8)

    traits = track_traits_during_collapse(model, n_traits=3)
    assert isinstance(traits, dict)
    assert 'survived' in traits
    assert 'lost' in traits
    assert 'amplified' in traits

    total_traits = len(traits['survived']) + len(traits['lost']) + len(traits['amplified'])
    assert total_traits == 3


def test_run_collapse_subliminal_experiment_mocked(monkeypatch):
    from experiments.collapse_subliminal import run_collapse_subliminal_experiment

    # Mock the slow training function
    def mock_train_and_evaluate(severity):
        return {
            'transfer_rate': severity * 0.1,
            'traits': {'survived': [0.3], 'lost': [0.1], 'amplified': [0.6]}
        }

    import experiments.collapse_subliminal
    monkeypatch.setattr(experiments.collapse_subliminal, "train_and_evaluate_collapse_transfer", mock_train_and_evaluate)

    results = run_collapse_subliminal_experiment([0.1, 0.2])
    assert len(results) == 2
    assert 0.1 in results
    assert 0.2 in results
    assert abs(results[0.1]['transfer_rate'] - 0.01) < 1e-5
    assert len(results[0.1]['traits']['survived']) == 1
