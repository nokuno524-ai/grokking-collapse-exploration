import os
import json
import pytest
import shutil
import tempfile
import torch
import hashlib
from typing import Any, Dict
from src.provenance import registry, lineage
from src.data import DatasetConfig, generate_modular_arithmetic

@pytest.fixture
def temp_registry():
    # Setup
    old_path = registry.REGISTRY_PATH
    fd, path = tempfile.mkstemp()
    os.close(fd)
    registry.REGISTRY_PATH = path
    # Make it valid empty json
    with open(path, 'w') as f:
        json.dump({}, f)

    yield path

    # Teardown
    registry.REGISTRY_PATH = old_path
    if os.path.exists(path):
        os.remove(path)

def test_register_and_query(temp_registry):
    entry = {
        "run_id": "test_1",
        "config_hash": "abc",
        "metrics": {"acc": 0.9}
    }
    registry.register("test_1", entry)

    res = registry.query({"run_id": "test_1"})
    assert len(res) == 1
    assert res[0]["metrics"]["acc"] == 0.9

    # Test duplicate ID
    with pytest.raises(ValueError):
        registry.register("test_1", {"run_id": "test_1"})

def test_registry_corrupt_json(temp_registry):
    with open(temp_registry, 'w') as f:
        f.write("invalid json")

    res = registry.query()
    assert res == []

def test_registry_diff(temp_registry):
    entry1 = {"run_id": "r1", "param": "A"}
    entry2 = {"run_id": "r2", "param": "B"}
    registry.register("r1", entry1)
    registry.register("r2", entry2)

    d = registry.diff("r1", "r2")
    assert "param" in d
    assert d["param"]["run_1"] == "A"
    assert d["param"]["run_2"] == "B"

def test_lineage_hashes():
    config = {"seed": 42, "prime": 59}
    hashes = lineage.generate_dataset_hashes(config)

    assert "train_in" in hashes
    assert "test_tgt" in hashes

    # Should verify correctly
    assert lineage.verify_dataset_lineage(config, hashes) is True

    # Modify a hash
    bad_hashes = dict(hashes)
    bad_hashes["train_in"] = "wrong_hash"
    assert lineage.verify_dataset_lineage(config, bad_hashes) is False

def test_hash_determinism():
    config = DatasetConfig(seed=42)
    t1, _, _, _ = generate_modular_arithmetic(config)
    h1 = lineage.compute_tensor_hash(t1)

    # Re-generate
    t2, _, _, _ = generate_modular_arithmetic(config)
    h2 = lineage.compute_tensor_hash(t2)

    assert h1 == h2
