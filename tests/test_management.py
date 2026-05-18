import pytest
import os
import json
import pandas as pd
from pathlib import Path
import tempfile
import yaml

from src.management.config import ExperimentConfig, ComputeConfig, ModelConfig, DatasetConfig, TrainingConfig
from src.management.slurm import SlurmGenerator
from src.management.results import ResultsCollector
from src.management.runner import build_tasks

def test_experiment_config_serialization():
    """Test converting config to dict and saving/loading YAML."""
    config = ExperimentConfig(
        name="test_exp",
        output_dir="test_out",
        seeds=[1, 2],
        weight_decays=[0.5],
        noise_fractions=[0.1, 0.2],
        compute=ComputeConfig(gpus=2, time="02:00:00")
    )

    # Check tasks calculation
    assert config.get_num_tasks() == 4  # 2 seeds * 1 wd * 2 noise * 1 c_level * 1 c_sev

    # Test dictionary conversion
    d = config.to_dict()
    assert d['name'] == "test_exp"
    assert d['compute']['gpus'] == 2
    assert d['dataset']['prime'] == 59  # Check default

    # Test yaml serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "config.yaml")
        config.save_yaml(yaml_path)

        loaded = ExperimentConfig.load_yaml(yaml_path)
        assert loaded.name == "test_exp"
        assert loaded.seeds == [1, 2]
        assert loaded.compute.time == "02:00:00"

def test_slurm_generator():
    """Test generating a Slurm script from config."""
    config = ExperimentConfig(
        name="test_slurm",
        output_dir="test_out",
        seeds=[1, 2, 3] # 3 tasks
    )

    generator = SlurmGenerator(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "job.sbatch")
        content = generator.generate_script(script_path)

        assert os.path.exists(script_path)
        assert "#SBATCH --job-name=test_slurm" in content
        assert "#SBATCH --array=0-2" in content
        assert "export PYTHONUNBUFFERED=1" in content
        assert "src/management/runner.py" in content
        assert "test_slurm_config.yaml" in content

def test_runner_build_tasks():
    """Test parameter grid expansion."""
    config = ExperimentConfig(
        name="test_grid",
        output_dir="out",
        seeds=[1, 2],
        weight_decays=[0.1],
        noise_fractions=[0.0, 0.1],
        collapse_levels=[0.0],
        collapse_severities=[0.5]
    )

    tasks = build_tasks(config)
    assert len(tasks) == 4
    # Expected order: wd, noise, c_level, c_sev, seed
    assert tasks[0] == (0.1, 0.0, 0.0, 0.5, 1)
    assert tasks[1] == (0.1, 0.0, 0.0, 0.5, 2)
    assert tasks[2] == (0.1, 0.1, 0.0, 0.5, 1)
    assert tasks[3] == (0.1, 0.1, 0.0, 0.5, 2)

def test_results_collector():
    """Test that results collector handles multiple JSON outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock structure
        root = Path(tmpdir)
        run1 = root / "run1"
        run2 = root / "run2"
        run1.mkdir()
        run2.mkdir()

        data1 = {
            "config": {"seed": 42, "weight_decay": 1.0},
            "grokked": True,
            "final_test_acc": 0.99,
            "data_hash": "hash123"
        }
        data2 = {
            "config": {"seed": 43, "weight_decay": 1.0},
            "grokked": False,
            "final_test_acc": 0.50,
            "data_hash": "hash456"
        }

        with open(run1 / "results.json", "w") as f:
            json.dump(data1, f)
        with open(run2 / "results.json", "w") as f:
            json.dump(data2, f)

        collector = ResultsCollector(str(root))
        df = collector.collect()

        assert not df.empty
        assert len(df) == 2
        assert 'config_seed' in df.columns
        assert 'grokked' in df.columns
        assert 'data_hash' in df.columns

        # Test exports
        csv_path = root / "out.csv"
        html_path = root / "out.html"
        collector.to_csv(df, str(csv_path))
        collector.to_html(df, str(html_path))

        assert csv_path.exists()
        assert html_path.exists()

        # Verify CSV has correct rows
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 2
