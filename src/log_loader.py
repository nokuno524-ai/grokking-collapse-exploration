import json
from pathlib import Path
from typing import Dict, Any

def load_results_json(run_dir: Path) -> Dict[str, Any]:
    """
    Load the results.json file from the given run directory.

    Args:
        run_dir: Path to the directory containing results.json

    Returns:
        Dictionary containing the parsed JSON data.

    Raises:
        FileNotFoundError: If results.json does not exist in the directory.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json found in {run_dir}")

    with open(results_path, "r") as f:
        return json.load(f)
