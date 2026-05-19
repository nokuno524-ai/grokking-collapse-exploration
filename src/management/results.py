import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


class ResultsCollector:
    """Recursively parses results.json files from output directories."""

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)

    def find_result_files(self) -> List[Path]:
        """Find all results.json files recursively."""
        return list(self.base_dir.rglob("results.json"))

    def load_result(self, path: Path) -> Dict[str, Any]:
        """Load a single result JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def aggregate_to_dataframe(self) -> pd.DataFrame:
        """Parse all found results and normalize them into a single pandas DataFrame."""
        result_files = self.find_result_files()
        all_data = []

        for path in result_files:
            try:
                data = self.load_result(path)

                # Flatten the 'config' dictionary into top-level keys for the DataFrame
                flattened_data = {}

                # Some old data formats might not have nested configs
                if 'config' in data:
                    # In our new format, config might be deeply nested
                    for section in ['model', 'dataset', 'training']:
                        if section in data['config']:
                            for k, v in data['config'][section].items():
                                flattened_data[f"{section}.{k}"] = v
                    # Extract experiment_name directly if present
                    if 'experiment_name' in data['config']:
                        flattened_data['experiment_name'] = data['config']['experiment_name']

                # Add top-level metrics
                for key in ['git_commit', 'grokked', 'grokking_step', 'final_train_acc',
                            'final_test_acc', 'final_weight_norm', 'final_embedding_rank',
                            'final_fourier_concentration']:
                    if key in data:
                        flattened_data[key] = data[key]

                # Keep path for reference
                flattened_data['path'] = str(path)

                all_data.append(flattened_data)
            except Exception as e:
                print(f"Error parsing {path}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)

    def load_history(self, path: Path) -> pd.DataFrame:
        """Load the history (training curve) of a single run as a DataFrame."""
        data = self.load_result(path)
        if 'history' in data:
            return pd.DataFrame(data['history'])
        return pd.DataFrame()
