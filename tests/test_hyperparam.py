import pytest
import os
from pathlib import Path
import yaml

def test_hyperparam_config_generation():
    # Import locally to test just the generation logic
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.hyperparam_sensitivity import SEVERITY_ORDER, WEIGHT_DECAYS, LEARNING_RATES, generate_configs, OUTPUT_DIR

    # Run config gen
    generate_configs()

    # Ensure a few files exist
    assert OUTPUT_DIR.exists()
    assert (OUTPUT_DIR / "pure").exists()

    file_path = OUTPUT_DIR / "pure" / f"config_wd{WEIGHT_DECAYS[0]}_lr{LEARNING_RATES[0]}.yaml"
    assert file_path.exists()

    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    assert config["condition"] == "pure"
    assert config["weight_decay"] == WEIGHT_DECAYS[0]
    assert config["learning_rate"] == LEARNING_RATES[0]
