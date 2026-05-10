import csv
import json
import jsonlines
import pathlib
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


def parse_csv_log(filepath: str) -> pd.DataFrame:
    """Parse CSV training logs with auto-detected columns."""
    return pd.read_csv(filepath)


def parse_jsonl_log(filepath: str) -> pd.DataFrame:
    """Parse JSON-lines format logs."""
    data = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            data.append(obj)
    return pd.DataFrame(data)


def scan_results_dir(results_dir: str) -> List[Dict[str, Any]]:
    """Catalog all result files with metadata."""
    results_path = pathlib.Path(results_dir)
    catalog = []

    if not results_path.exists() or not results_path.is_dir():
        return catalog

    for condition_dir in results_path.iterdir():
        if not condition_dir.is_dir():
            continue

        results_file = condition_dir / "results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file, "r") as f:
                data = json.load(f)

            config = data.get("config", {})
            catalog.append({
                "condition_name": condition_dir.name,
                "path": str(condition_dir),
                "results_file": str(results_file),
                "has_history": "history" in data and len(data["history"]) > 0,
                "grokked": data.get("grokked", False),
                "final_test_acc": data.get("final_test_acc", 0.0),
                "collapse_level": config.get("collapse_level", 0.0),
            })
        except Exception as e:
            print(f"Error reading {results_file}: {e}")

    return catalog


def load_experiment(name: str, results_dir: str = "results") -> Dict[str, Any]:
    """Load all logs for a named experiment."""
    results_path = pathlib.Path(results_dir) / name / "results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Experiment results not found at {results_path}")

    with open(results_path, "r") as f:
        data = json.load(f)

    return data


def detect_grokking_point(df: pd.DataFrame, acc_col: str = "val_acc", threshold: float = 0.9) -> int:
    """Find step where accuracy crosses threshold."""
    if acc_col not in df.columns:
        raise ValueError(f"Column {acc_col} not found in dataframe")

    if "step" not in df.columns:
        raise ValueError("Column 'step' not found in dataframe")

    crossing_points = df[df[acc_col] >= threshold]
    if len(crossing_points) == 0:
        return -1

    return int(crossing_points.iloc[0]["step"])


def compute_collapse_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute weight norm reduction, representation rank, gradient norms."""
    metrics = {}

    # Weight norm reduction (from start to end)
    if "weight_norm" in df.columns and len(df) >= 2:
        start_norm = float(df["weight_norm"].iloc[0])
        end_norm = float(df["weight_norm"].iloc[-1])
        if start_norm > 0:
            reduction = (start_norm - end_norm) / start_norm
            metrics["weight_norm_reduction"] = reduction
        else:
            metrics["weight_norm_reduction"] = 0.0

    # Final representation rank
    if "embedding_rank" in df.columns and len(df) > 0:
        metrics["final_representation_rank"] = float(df["embedding_rank"].iloc[-1])

    # Gradient norm (if available)
    if "grad_norm" in df.columns and len(df) > 0:
        metrics["final_grad_norm"] = float(df["grad_norm"].iloc[-1])
        metrics["mean_grad_norm"] = float(df["grad_norm"].mean())

    return metrics
