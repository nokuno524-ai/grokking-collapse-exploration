import json
import pytest
from pathlib import Path
from src.provenance.registry import ExperimentRegistry

def test_log_and_query_runs(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ExperimentRegistry(filepath=registry_path)

    config1 = {"param1": 1, "param2": "A"}
    config2 = {"param1": 2, "param2": "B"}

    registry.log_run(config1, 42, [{"step": 1, "loss": 0.5}], "completed")
    registry.log_run(config2, 42, [{"step": 1, "loss": 0.3}], "completed")

    all_runs = registry.query_runs()
    assert len(all_runs) == 2

    filtered_runs = registry.query_runs({"param1": 1})
    assert len(filtered_runs) == 1
    assert filtered_runs[0]["config"]["param2"] == "A"
    assert filtered_runs[0]["status"] == "completed"

def test_integrity(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ExperimentRegistry(filepath=registry_path)

    # Write invalid json
    with open(registry_path, "w") as f:
        f.write("invalid json")

    runs = registry.query_runs()
    assert runs == []
