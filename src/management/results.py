import os
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

class ResultsCollector:
    """Collects and aggregates experiment results from output directories."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def collect(self) -> pd.DataFrame:
        """Scan directories and build a DataFrame of all results."""
        records = []
        if not self.root_dir.exists():
            print(f"Warning: Root directory {self.root_dir} does not exist.")
            return pd.DataFrame()

        # Look for all results.json files in subdirectories
        for result_file in self.root_dir.rglob("results.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                # Flatten the data structure
                record = {}

                # Extract config
                config = data.get("config", {})
                for k, v in config.items():
                    record[f"config_{k}"] = v

                # Extract main results
                for k in ["grokked", "grokking_step", "final_train_acc", "final_test_acc",
                          "final_weight_norm", "final_embedding_rank", "final_fourier_concentration",
                          "data_hash", "git_commit"]:
                    if k in data:
                        record[k] = data[k]

                records.append(record)
            except Exception as e:
                print(f"Error reading {result_file}: {e}")

        return pd.DataFrame(records)

    def to_csv(self, df: pd.DataFrame, out_path: str):
        """Save results to CSV."""
        if df.empty:
            print("DataFrame is empty, not saving CSV.")
            return

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")

    def to_html(self, df: pd.DataFrame, out_path: str):
        """Save results to HTML format."""
        if df.empty:
            print("DataFrame is empty, not saving HTML.")
            return

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # Format HTML with some basic styling
        html = f"""
        <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h2>Experiment Results</h2>
            {df.to_html(index=False, classes='table table-striped')}
        </body>
        </html>
        """

        with open(out_path, "w") as f:
            f.write(html)
        print(f"Saved HTML results to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="results")
    parser.add_argument("--out-csv", type=str, default="analysis/aggregated_results.csv")
    parser.add_argument("--out-html", type=str, default="analysis/aggregated_results.html")
    args = parser.parse_args()

    collector = ResultsCollector(args.dir)
    df = collector.collect()

    if not df.empty:
        print(f"Found {len(df)} results")
        collector.to_csv(df, args.out_csv)
        collector.to_html(df, args.out_html)
    else:
        print("No results found.")
