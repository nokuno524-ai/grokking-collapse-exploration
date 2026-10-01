from typing import Tuple
import json
import os
from typing import Dict, Any

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', 'registry.json')

def load_registry() -> list:
    """Loads the results registry."""
    if not os.path.exists(REGISTRY_PATH):
        return []
    try:
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_registry(registry: list) -> None:
    """Saves the results registry."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)

def register_run(config_hash: str, metrics: Dict[str, Any], effect_sizes: Dict[str, float], cis: Dict[str, Tuple[float, float]]) -> None:
    """
    Registers a single run's results into the registry.
    """
    registry = load_registry()

    # Remove if exists to update
    registry = [r for r in registry if r.get('config_hash') != config_hash]

    entry = {
        'config_hash': config_hash,
        'metrics': metrics,
        'effect_sizes': effect_sizes,
        'CIs': cis
    }

    registry.append(entry)
    save_registry(registry)
