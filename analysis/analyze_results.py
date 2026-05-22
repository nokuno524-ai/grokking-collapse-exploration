import json
import os
import glob
import pandas as pd
import numpy as np
import scipy.stats
from pathlib import Path
from typing import List, Dict, Any

def load_all_results(base_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Finds all results.json files under base_dir and loads them into a list of dicts.
    """
    base_path = Path(base_dir)
    results_files = list(base_path.rglob("results.json"))

    results_list = []
    for filepath in results_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            config = data.get("config", {})
            condition_name = config.get("condition_name", filepath.parent.name)

            # Use 'pure' if the condition_name is not present, etc, just a fallback
            if condition_name == "":
                condition_name = "unknown"

            # Keep flat structure for easy dataframe conversion
            flat_data = {
                "condition": condition_name,
                "seed": config.get("seed", 0),
                "filepath": str(filepath),
                "grokked": data.get("grokked", False),
                "grokking_step": data.get("grokking_step", -1),
                "final_train_acc": data.get("final_train_acc", 0.0),
                "final_test_acc": data.get("final_test_acc", 0.0),
                "final_weight_norm": data.get("final_weight_norm", 0.0),
                "final_embedding_rank": data.get("final_embedding_rank", 0.0),
                "final_fourier_concentration": data.get("final_fourier_concentration", 0.0),
            }

            # Optional: history
            if "history" in data:
                flat_data["history"] = data["history"]

            results_list.append(flat_data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return results_list

def build_dataframe(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts list of dicts to pandas DataFrame.
    Drops 'history' for the main summary dataframe.
    """
    clean_list = []
    for r in results_list:
        clean = {k: v for k, v in r.items() if k != "history"}
        clean_list.append(clean)

    df = pd.DataFrame(clean_list)
    return df

def generate_summary_tables(df: pd.DataFrame, output_path: Path):
    """
    Aggregates mean/std of metrics by condition and outputs markdown.
    """
    metrics = [
        "final_train_acc",
        "final_test_acc",
        "final_weight_norm",
        "final_embedding_rank",
        "final_fourier_concentration",
        "grokking_step"
    ]

    # Filter df to keep only these metrics plus condition
    available_metrics = [m for m in metrics if m in df.columns]

    summary = df.groupby("condition")[available_metrics].agg(["mean", "std", "count"])

    # Format to markdown
    md_str = "# Experiment Summary Table\n\n"

    # Flatten multi-index columns for easier reading
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    md_str += summary.to_markdown()

    with open(output_path, "w") as f:
        f.write(md_str)

    print(f"Summary table written to {output_path}")
    return summary

def compute_statistical_tests(df: pd.DataFrame, output_path: Path):
    """
    Uses scipy.stats.bootstrap to compute CIs for differences and output markdown.
    Compares each condition to 'pure' if available.
    """
    md_str = "# Statistical Tests\n\n"

    pure_df = df[df["condition"] == "pure"]
    if pure_df.empty:
        md_str += "No 'pure' condition found for baseline comparison.\n"
        with open(output_path, "w") as f:
            f.write(md_str)
        return

    conditions = df["condition"].unique()
    metrics = ["final_test_acc", "final_fourier_concentration", "grokking_step"]
    available_metrics = [m for m in metrics if m in df.columns]

    for metric in available_metrics:
        md_str += f"## Metric: {metric}\n\n"
        md_str += "| Condition | Difference from Pure (Mean) | 95% CI (Bootstrap) | Mann-Whitney p-value |\n"
        md_str += "|---|---|---|---|\n"

        pure_vals = pure_df[metric].dropna().values
        if len(pure_vals) < 2:
            md_str += f"Not enough data in 'pure' for {metric}.\n"
            continue

        for cond in conditions:
            if cond == "pure":
                continue

            cond_vals = df[df["condition"] == cond][metric].dropna().values
            if len(cond_vals) < 2:
                md_str += f"| {cond} | Not enough data | N/A | N/A |\n"
                continue

            mean_diff = np.mean(cond_vals) - np.mean(pure_vals)

            # Bootstrap CI for the difference in means
            # Define statistic function properly handling axis
            def diff_of_means(x, y, axis=-1):
                return np.mean(x, axis=axis) - np.mean(y, axis=axis)

            try:
                res = scipy.stats.bootstrap(
                    (cond_vals, pure_vals),
                    statistic=diff_of_means,
                    vectorized=True,
                    axis=-1,
                    n_resamples=1000,
                    confidence_level=0.95,
                    method='BCa'
                )
                ci_low, ci_high = res.confidence_interval
                ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"
            except Exception as e:
                ci_str = f"Error: {e}"

            # Mann-Whitney U test
            try:
                _, p_val = scipy.stats.mannwhitneyu(cond_vals, pure_vals, alternative='two-sided')
                p_str = f"{p_val:.4e}"
            except Exception as e:
                p_str = f"Error"

            md_str += f"| {cond} | {mean_diff:.3f} | {ci_str} | {p_str} |\n"

        md_str += "\n"

    with open(output_path, "w") as f:
        f.write(md_str)

    print(f"Statistical tests written to {output_path}")


if __name__ == "__main__":
    results_dir = Path("results")
    if results_dir.exists():
        results_list = load_all_results(results_dir)
        print(f"Loaded {len(results_list)} results.")
        df = build_dataframe(results_list)
        print(f"Built dataframe with shape: {df.shape}")

        output_dir = Path("analysis")
        output_dir.mkdir(exist_ok=True)

        generate_summary_tables(df, output_dir / "summary_report.md")
        compute_statistical_tests(df, output_dir / "statistical_tests.md")
