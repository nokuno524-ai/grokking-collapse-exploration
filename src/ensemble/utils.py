"""
Utility functions for ensemble runners and aggregation.
"""

import os
import json
from pathlib import Path
from typing import Iterator, Dict, Any

def load_run_results(base_dir: str) -> Iterator[Dict[str, Any]]:
    """
    Traverse directory trees to find results.json files,
    parsing metadata and yielding result dicts.
    Ignores missing or corrupted JSON files.

    Expected structure: base_dir/<seed>/<condition>/results.json
    or similar nested directories.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return

    for json_path in base_path.rglob("results.json"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            # Add path info to identify the run
            data["_source_path"] = str(json_path)
            yield data
        except (json.JSONDecodeError, OSError) as e:
            # Skip corrupted or unreadable files
            print(f"Warning: Failed to read {json_path}: {e}")
            continue
