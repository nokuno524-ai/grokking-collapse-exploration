import pytest
import tempfile
import os
from src.config import ExperimentConfig, load_config

def test_load_config():
    valid_yaml = """
task: sparse_parity
prime: 11
d_model: 64
"""
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(valid_yaml)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config.task == "sparse_parity"
        assert config.prime == 11
        assert config.d_model == 64
        # Check default fallback
        assert config.n_heads == 4
    finally:
        os.remove(temp_path)

def test_load_config_invalid_task():
    invalid_yaml = """
task: invalid_task
prime: 11
"""
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(invalid_yaml)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid task"):
            load_config(temp_path)
    finally:
        os.remove(temp_path)
