import pytest
from experiments.scaling import ScalingExperimentConfig

def test_scaling_config_defaults():
    config = ScalingExperimentConfig()

    # Check default values
    assert config.d_models == [32, 64, 128, 256, 512]
    assert config.primes == [29, 59, 97, 113, 127]
    assert len(config.collapse_levels) == 10
    assert 0.0 in config.collapse_levels
    assert 0.5 in config.collapse_levels
    assert config.max_steps == 10000
    assert config.eval_every == 500


def test_scaling_config_custom():
    config = ScalingExperimentConfig(
        d_models=[16, 32],
        primes=[11],
        collapse_levels=[0.1, 0.2, 0.3],
        max_steps=5000,
        eval_every=100
    )

    assert config.d_models == [16, 32]
    assert config.primes == [11]
    assert config.collapse_levels == [0.1, 0.2, 0.3]
    assert config.max_steps == 5000
    assert config.eval_every == 100
