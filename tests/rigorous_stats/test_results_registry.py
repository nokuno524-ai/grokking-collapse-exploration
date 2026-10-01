import pytest
import os
import json
from src.rigorous_stats.results_registry import register_run, load_registry, REGISTRY_PATH

def test_results_registry(tmpdir, monkeypatch):
    test_registry = os.path.join(tmpdir, 'registry.json')
    monkeypatch.setattr('src.rigorous_stats.results_registry.REGISTRY_PATH', test_registry)

    # initially empty
    assert load_registry() == []

    register_run(
        config_hash="abc",
        metrics={'acc': 0.9},
        effect_sizes={'d': 1.0},
        cis={'d': [0.5, 1.5]}
    )

    reg = load_registry()
    assert len(reg) == 1
    assert reg[0]['config_hash'] == "abc"

    # updating replaces it
    register_run(
        config_hash="abc",
        metrics={'acc': 0.95},
        effect_sizes={'d': 1.0},
        cis={'d': [0.5, 1.5]}
    )
    reg2 = load_registry()
    assert len(reg2) == 1
    assert reg2[0]['metrics']['acc'] == 0.95
