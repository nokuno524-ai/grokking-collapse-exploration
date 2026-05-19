import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def collect_results(base_dir: str) -> pd.DataFrame:
    """
    Recursively scans base_dir for results.json files and aggregates them
    into a single Pandas DataFrame.
    """
    records = []
    base_path = Path(base_dir)

    for path in base_path.rglob("results.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Flatten config
            config_data = flatten_dict(data.get("config", {}))

            # Extract top-level metrics
            metrics = {
                "grokked": data.get("grokked", False),
                "grokking_step": data.get("grokking_step", None),
                "final_train_acc": data.get("final_train_acc", 0.0),
                "final_test_acc": data.get("final_test_acc", 0.0),
                "final_weight_norm": data.get("final_weight_norm", 0.0),
                "final_fourier_concentration": data.get("final_fourier_concentration", 0.0),
                "source_path": str(path.parent)
            }

            # Combine
            record = {**config_data, **metrics}
            records.append(record)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return pd.DataFrame(records)

def plot_condition_comparison(df: pd.DataFrame, output_dir: str):
    """Generates summary plots comparing different experimental conditions."""
    os.makedirs(output_dir, exist_ok=True)

    # Needs a column to group by, default to 'condition_name' if available,
    # else construct one from noise and wd
    if 'condition_name' not in df.columns:
        if 'noise_fraction' in df.columns and 'weight_decay' in df.columns:
            df['condition_name'] = df.apply(
                lambda row: f"wd={row['weight_decay']}_n={row['noise_fraction']}", axis=1
            )
        else:
            df['condition_name'] = 'unknown'

    # 1. Final Test Accuracy Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='condition_name', y='final_test_acc')
    plt.xticks(rotation=45, ha='right')
    plt.title('Final Test Accuracy by Condition')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_acc_comparison.png"), dpi=300)
    plt.close()

    # 2. Grokking Step (for those that grokked)
    grokked_df = df[df['grokked'] == True]
    if not grokked_df.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=grokked_df, x='condition_name', y='grokking_step')
        plt.xticks(rotation=45, ha='right')
        plt.title('Grokking Step by Condition (Successful Runs Only)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "grokking_step_comparison.png"), dpi=300)
        plt.close()

    # 3. Fourier Concentration
    if 'final_fourier_concentration' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='condition_name', y='final_fourier_concentration')
        plt.xticks(rotation=45, ha='right')
        plt.title('Final Fourier Concentration by Condition')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fourier_concentration_comparison.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="analysis/summary")
    parser.add_argument("--output-csv", type=str, default="analysis/summary/aggregated_results.csv")
    args = parser.parse_args()

    df = collect_results(args.results_dir)
    print(f"Found {len(df)} result files.")

    if not df.empty:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved aggregated results to {args.output_csv}")

        plot_condition_comparison(df, args.output_dir)
        print(f"Saved comparison plots to {args.output_dir}")
    else:
        print("No results found to aggregate.")
