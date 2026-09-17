import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

class ExperimentRegistry:
    def __init__(self, filepath: str | Path = "experiments/registry.json"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        if not self.filepath.exists():
            with open(self.filepath, "w") as f:
                json.dump([], f)

    def _read_db(self) -> List[Dict]:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _write_db(self, data: List[Dict]) -> None:
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            return "unknown"

    def _hash_config(self, config: Dict[str, Any]) -> str:
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def log_run(self, config: Dict[str, Any], seed: int, metrics_timeline: List[Dict], status: str) -> str:
        """Logs a single training run to the registry."""
        data = self._read_db()
        config_hash = self._hash_config(config)
        git_commit = self._get_git_commit()

        run_record = {
            "config_hash": config_hash,
            "config": config,
            "seed": seed,
            "git_commit": git_commit,
            "metrics_timeline": metrics_timeline,
            "status": status,
        }

        data.append(run_record)
        self._write_db(data)
        return config_hash

    def query_runs(self, config_filter: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Queries runs based on matching config keys."""
        data = self._read_db()
        if not config_filter:
            return data

        filtered = []
        for run in data:
            match = True
            for k, v in config_filter.items():
                if run.get("config", {}).get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(run)
        return filtered
