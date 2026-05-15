"""
Statistical analysis tools for experiment results.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """Analyzer for experiment results to compute statistics and significance."""

    def __init__(self):
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def load_results(self, results_dir: Path) -> None:
        """Load experiment results from subdirectories in the given directory."""
        if not results_dir.exists():
            logger.warning(f"Directory {results_dir} does not exist.")
            return

        for condition_dir in results_dir.iterdir():
            if not condition_dir.is_dir():
                continue

            condition_name = condition_dir.name
            if condition_name not in self.results:
                self.results[condition_name] = []

            # Check for seed subdirectories
            has_seeds = False
            for seed_dir in condition_dir.iterdir():
                if seed_dir.is_dir() and (seed_dir / "results.json").exists():
                    has_seeds = True
                    try:
                        with open(seed_dir / "results.json", "r") as f:
                            data = json.load(f)
                            self.results[condition_name].append(data)
                    except Exception as e:
                        logger.error(f"Error loading {seed_dir / 'results.json'}: {e}")

            # If no seed subdirectories, check the condition dir itself
            if not has_seeds and (condition_dir / "results.json").exists():
                try:
                    with open(condition_dir / "results.json", "r") as f:
                        data = json.load(f)
                        self.results[condition_name].append(data)
                except Exception as e:
                    logger.error(f"Error loading {condition_dir / 'results.json'}: {e}")

    def compute_summary_statistics(self, metrics: List[str] = None) -> pd.DataFrame:
        """Compute mean, std, median, and IQR for specified metrics grouped by condition."""
        if metrics is None:
            metrics = ["final_train_acc", "final_test_acc", "final_weight_norm", "grokking_step"]

        summary_data = []
        for condition, runs in self.results.items():
            if not runs:
                continue

            condition_stats = {"condition": condition, "n_runs": len(runs)}

            for metric in metrics:
                values = [r.get(metric) for r in runs if r.get(metric) is not None]
                if not values:
                    continue

                condition_stats[f"{metric}_mean"] = np.mean(values)
                condition_stats[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                condition_stats[f"{metric}_median"] = np.median(values)
                q75, q25 = np.percentile(values, [75 ,25])
                condition_stats[f"{metric}_iqr"] = q75 - q25

            summary_data.append(condition_stats)

        return pd.DataFrame(summary_data)

    def test_statistical_significance(
        self, condition1: str, condition2: str, metric: str, bootstrap_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Perform statistical tests (t-test, Mann-Whitney U, bootstrap CI)
        between two conditions for a specific metric.
        """
        if condition1 not in self.results or condition2 not in self.results:
            raise ValueError("One or both conditions not found in loaded results.")

        v1 = [r.get(metric) for r in self.results[condition1] if r.get(metric) is not None]
        v2 = [r.get(metric) for r in self.results[condition2] if r.get(metric) is not None]

        if not v1 or not v2:
            raise ValueError(f"No valid data for metric {metric} in one or both conditions.")

        results = {}

        # T-test
        if len(v1) > 1 and len(v2) > 1:
            t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
            results["ttest"] = {"statistic": float(t_stat), "p_value": float(p_val)}

        # Mann-Whitney U test
        u_stat, u_p_val = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        results["mann_whitney"] = {"statistic": float(u_stat), "p_value": float(u_p_val)}

        # Bootstrap CI for the difference of means
        def diff_of_means(data1, data2, axis=-1):
            return np.mean(data1, axis=axis) - np.mean(data2, axis=axis)

        if len(v1) > 1 and len(v2) > 1:
            res = stats.bootstrap((v1, v2), diff_of_means, n_resamples=bootstrap_samples, method="BCa")
            results["bootstrap_ci"] = {
                "low": float(res.confidence_interval.low),
                "high": float(res.confidence_interval.high),
                "mean_diff": float(np.mean(v1) - np.mean(v2))
            }

        return results

    def compute_effect_sizes(self, condition1: str, condition2: str, metric: str) -> float:
        """Compute Cohen's d for the specified metric between two conditions."""
        if condition1 not in self.results or condition2 not in self.results:
            raise ValueError("One or both conditions not found in loaded results.")

        v1 = [r.get(metric) for r in self.results[condition1] if r.get(metric) is not None]
        v2 = [r.get(metric) for r in self.results[condition2] if r.get(metric) is not None]

        if len(v1) < 2 or len(v2) < 2:
            return 0.0

        mean1, mean2 = np.mean(v1), np.mean(v2)
        var1, var2 = np.var(v1, ddof=1), np.var(v2, ddof=1)
        n1, n2 = len(v1), len(v2)

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        if pooled_var == 0:
            return 0.0

        cohens_d = (mean1 - mean2) / np.sqrt(pooled_var)
        return float(cohens_d)

    def generate_latex_table(self, metrics: List[str] = None) -> str:
        """Generate a LaTeX table summarizing the results."""
        df = self.compute_summary_statistics(metrics)
        if df.empty:
            return "No data available."

        # Format the dataframe for LaTeX
        latex_str = "\\begin{table}[h]\n\\centering\n"

        # Create column format string e.g. "l|c|c|c"
        cols = ["l"] + ["c"] * (len(df.columns) - 1)
        latex_str += f"\\begin{{tabular}}{{{'|'.join(cols)}}}\n"

        # Header
        headers = [col.replace("_", "\\_") for col in df.columns]
        latex_str += " & ".join(headers) + " \\\\\n\\hline\n"

        # Rows
        for _, row in df.iterrows():
            row_vals = []
            for val in row:
                if isinstance(val, float):
                    row_vals.append(f"{val:.3f}")
                else:
                    row_vals.append(str(val).replace("_", "\\_"))
            latex_str += " & ".join(row_vals) + " \\\\\n"

        latex_str += "\\end{tabular}\n"
        latex_str += "\\caption{Summary Statistics}\n"
        latex_str += "\\end{table}"

        return latex_str
