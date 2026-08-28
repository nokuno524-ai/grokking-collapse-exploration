import json
import pytest
from pathlib import Path
from src.log_loader import load_results_json

def test_load_results_json_success(tmp_path):
    mock_data = {"grokked": True, "history": [{"step": 100}]}
    results_file = tmp_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(mock_data, f)

    loaded_data = load_results_json(tmp_path)
    assert loaded_data == mock_data
    assert loaded_data["grokked"] is True
    assert loaded_data["history"][0]["step"] == 100

def test_load_results_json_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No results.json found in"):
        load_results_json(tmp_path)

def test_load_results_json_invalid(tmp_path):
    results_file = tmp_path / "results.json"
    with open(results_file, "w") as f:
        f.write("{invalid json")

    with pytest.raises(json.JSONDecodeError):
        load_results_json(tmp_path)
