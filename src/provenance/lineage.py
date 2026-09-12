import hashlib
import torch
import json
from dataclasses import asdict
from typing import Dict, Any

from src.data import generate_modular_arithmetic, DatasetConfig

def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """Compute deterministic SHA-256 hash of a tensor's data."""
    if tensor is None:
        return ""
    # Ensure contiguous and CPU for consistent hashing
    t_data = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(t_data.tobytes()).hexdigest()

def verify_dataset_lineage(config_dict: Dict[str, Any], expected_hashes: Dict[str, str]) -> bool:
    """
    Given a dataset config and expected hashes, regenerate the dataset and verify hashes match.
    """
    # Create config object (filter only valid fields)
    valid_keys = DatasetConfig.__dataclass_fields__.keys()
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = DatasetConfig(**filtered_config)

    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    current_hashes = {
        "train_in": compute_tensor_hash(train_in),
        "train_tgt": compute_tensor_hash(train_tgt),
        "test_in": compute_tensor_hash(test_in),
        "test_tgt": compute_tensor_hash(test_tgt),
    }

    for key, expected_hash in expected_hashes.items():
        if current_hashes.get(key) != expected_hash:
            print(f"Hash mismatch for {key}: expected {expected_hash}, got {current_hashes.get(key)}")
            return False

    return True

def generate_dataset_hashes(config_dict: Dict[str, Any]) -> Dict[str, str]:
    """Generate hashes for a dataset config."""
    valid_keys = DatasetConfig.__dataclass_fields__.keys()
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = DatasetConfig(**filtered_config)

    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    return {
        "train_in": compute_tensor_hash(train_in),
        "train_tgt": compute_tensor_hash(train_tgt),
        "test_in": compute_tensor_hash(test_in),
        "test_tgt": compute_tensor_hash(test_tgt),
    }

def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute hash of config dict (sorted keys for determinism)."""
    # Exclude non-determinative or dynamic keys if needed, but for now hash all
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
