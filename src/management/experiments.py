"""
Experiment Manager for tracking and comparing runs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages experiment metadata, comparisons, and run filtering."""

    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def track_experiment(self, exp_id: str, metadata: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Track a new experiment with its metadata and results."""
        self.experiments[exp_id] = {
            "metadata": metadata,
            "results": results
        }

    def load_from_directory(self, base_dir: Path) -> None:
        """Load multiple experiments from a structured results directory."""
        if not base_dir.exists():
            logger.warning(f"Directory {base_dir} does not exist.")
            return

        for condition_dir in base_dir.iterdir():
            if not condition_dir.is_dir():
                continue

            condition_name = condition_dir.name

            for seed_dir in condition_dir.iterdir():
                if seed_dir.is_dir() and (seed_dir / "results.json").exists():
                    exp_id = f"{condition_name}_{seed_dir.name}"
                    try:
                        with open(seed_dir / "results.json", "r") as f:
                            data = json.load(f)
                            config = data.get("config", {})
                            results = {k: v for k, v in data.items() if k != "config" and k != "history"}
                            # Track history separately if needed, or keep it in results
                            if "history" in data:
                                results["history"] = data["history"]

                            self.track_experiment(exp_id, config, results)
                    except Exception as e:
                        logger.error(f"Error loading {seed_dir / 'results.json'}: {e}")

    def compare_experiments(self, exp_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments side-by-side."""
        comparison_data = []
        for exp_id in exp_ids:
            if exp_id not in self.experiments:
                logger.warning(f"Experiment {exp_id} not found.")
                continue

            exp = self.experiments[exp_id]
            flat_dict = {"exp_id": exp_id}

            # Flatten metadata
            for k, v in exp["metadata"].items():
                flat_dict[f"config_{k}"] = v

            # Flatten results (excluding history for the table)
            for k, v in exp["results"].items():
                if k != "history":
                    flat_dict[f"result_{k}"] = v

            comparison_data.append(flat_dict)

        return pd.DataFrame(comparison_data)

    def generate_experiment_card(self, exp_id: str) -> str:
        """Generate a 1-page summary text for a specific experiment."""
        if exp_id not in self.experiments:
            return f"Experiment {exp_id} not found."

        exp = self.experiments[exp_id]
        meta = exp["metadata"]
        res = exp["results"]

        card = f"=== Experiment Card: {exp_id} ===\n\n"

        card += "Hyperparameters & Config:\n"
        card += "-" * 25 + "\n"
        for k, v in meta.items():
            card += f"  {k}: {v}\n"

        card += "\nKey Results:\n"
        card += "-" * 25 + "\n"
        card += f"  Grokked: {res.get('grokked', False)}\n"
        card += f"  Grokking Step: {res.get('grokking_step', 'N/A')}\n"
        card += f"  Final Test Acc: {res.get('final_test_acc', 'N/A'):.4f}\n" if isinstance(res.get('final_test_acc'), (int, float)) else f"  Final Test Acc: {res.get('final_test_acc', 'N/A')}\n"
        card += f"  Final Train Acc: {res.get('final_train_acc', 'N/A'):.4f}\n" if isinstance(res.get('final_train_acc'), (int, float)) else f"  Final Train Acc: {res.get('final_train_acc', 'N/A')}\n"
        card += f"  Final Weight Norm: {res.get('final_weight_norm', 'N/A'):.2f}\n" if isinstance(res.get('final_weight_norm'), (int, float)) else f"  Final Weight Norm: {res.get('final_weight_norm', 'N/A')}\n"

        history = res.get("history", [])
        if history:
            card += f"\nTraining Duration: {len(history)} logged steps\n"
            card += f"Max Step Reached: {history[-1].get('step', 'Unknown')}\n"

        return card

    def find_best_worst_runs(self, metric: str, maximize: bool = True, top_k: int = 1) -> List[str]:
        """Find the best or worst runs based on a specific metric."""
        valid_runs = []
        for exp_id, exp in self.experiments.items():
            val = exp["results"].get(metric)
            if val is not None:
                valid_runs.append((exp_id, val))

        if not valid_runs:
            return []

        sorted_runs = sorted(valid_runs, key=lambda x: x[1], reverse=maximize)
        return [run[0] for run in sorted_runs[:top_k]]
