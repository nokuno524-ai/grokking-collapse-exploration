import os
from src.experiments.config import ExperimentConfig, CollapseConfig, save_config, load_config, validate_config

def test_config_creation():
    config = ExperimentConfig(collapse_level=0.5)
    assert config.collapse_level == 0.5
    assert config.collapse_config is not None
    assert config.collapse_config.severity == "severe"

def test_config_validation():
    # Valid config
    config = ExperimentConfig(collapse_level=0.5)
    errors = validate_config(config)
    assert len(errors) == 0

    # Invalid collapse level
    config_invalid_level = ExperimentConfig(collapse_level=1.5)
    errors = validate_config(config_invalid_level)
    assert len(errors) > 0
    assert any("collapse_level" in e for e in errors)

    # Invalid collapse config
    collapse = CollapseConfig(collapse_type="invalid_type", severity="invalid_sev", injection_point="invalid_inj")
    config_invalid_collapse = ExperimentConfig(collapse_config=collapse)
    errors = validate_config(config_invalid_collapse)
    assert len(errors) == 3

def test_save_load_roundtrip(tmp_path):
    config = ExperimentConfig(collapse_level=0.2, learning_rate=0.005)

    # Test JSON
    json_path = tmp_path / "config.json"
    save_config(config, str(json_path))
    loaded_json = load_config(str(json_path))
    assert loaded_json.collapse_level == 0.2
    assert loaded_json.learning_rate == 0.005
    assert loaded_json.collapse_config.severity == "high" # since 0.2 > 0.15

    # Test YAML
    yaml_path = tmp_path / "config.yaml"
    save_config(config, str(yaml_path))
    loaded_yaml = load_config(str(yaml_path))
    assert loaded_yaml.collapse_level == 0.2
    assert loaded_yaml.learning_rate == 0.005
