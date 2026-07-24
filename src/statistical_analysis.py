import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
import pandas as pd

def load_results(json_path: str = "parsed_results.json") -> pd.DataFrame:
    """Load parsed results into a pandas DataFrame."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def compute_confidence_intervals(df: pd.DataFrame, metric: str, group_by: str = "config_condition_name", confidence: float = 0.95) -> pd.DataFrame:
    """Compute confidence intervals for a metric across groups."""

    # Filter out NaNs for the metric
    valid_df = df.dropna(subset=[metric])

    results = []
    for name, group in valid_df.groupby(group_by):
        values = group[metric].values
        n = len(values)

        if n == 0:
            continue

        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0

        if n > 1:
            # t-distribution critical value
            h = std * stats.t.ppf((1 + confidence) / 2., n-1) / np.sqrt(n)
        else:
            h = 0

        results.append({
            "Condition": name,
            "N": n,
            "Mean": mean,
            "Std": std,
            "CI_Lower": mean - h,
            "CI_Upper": mean + h,
            "CI_Margin": h
        })

    return pd.DataFrame(results).sort_values("Mean", ascending=False)

def bootstrap_ci(data: np.ndarray, num_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean."""
    data = np.asarray(data)
    n = len(data)

    if n == 0:
        return np.nan, np.nan, np.nan

    if n == 1:
        return data[0], data[0], data[0]

    # Generate bootstrap samples
    indices = np.random.randint(0, n, (num_bootstrap, n))
    samples = data[indices]

    # Compute statistic (mean) for each sample
    stat_dist = np.mean(samples, axis=1)

    # Compute percentiles
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100

    lower_bound = np.percentile(stat_dist, lower_percentile)
    upper_bound = np.percentile(stat_dist, upper_percentile)
    mean_val = np.mean(stat_dist)

    return mean_val, lower_bound, upper_bound

def mann_whitney_u_test(df: pd.DataFrame, metric: str, group_by: str = "config_condition_name", baseline_group: str = "pure") -> pd.DataFrame:
    """Perform Mann-Whitney U test comparing conditions to a baseline."""

    valid_df = df.dropna(subset=[metric, group_by])

    baseline_data = valid_df[valid_df[group_by] == baseline_group][metric].values

    if len(baseline_data) == 0:
        return pd.DataFrame()

    results = []
    for name, group in valid_df.groupby(group_by):
        if name == baseline_group:
            continue

        compare_data = group[metric].values

        if len(compare_data) == 0:
            continue

        try:
            stat, p_value = stats.mannwhitneyu(baseline_data, compare_data, alternative='two-sided')

            # Effect size (rank biserial correlation)
            n1, n2 = len(baseline_data), len(compare_data)
            expected_u = n1 * n2 / 2
            std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z = (stat - expected_u) / std_u if std_u > 0 else 0
            r = z / np.sqrt(n1 + n2) if (n1 + n2) > 0 else 0

            results.append({
                "Baseline": baseline_group,
                "Comparison": name,
                "U_Stat": stat,
                "P_Value": p_value,
                "Effect_Size_r": r,
                "N1": n1,
                "N2": n2,
                "Significant_05": p_value < 0.05
            })
        except Exception as e:
            print(f"Error testing {name} vs {baseline_group}: {e}")

    return pd.DataFrame(results).sort_values("P_Value")

def compute_correlations(df: pd.DataFrame, col1: str = "config_collapse_level", col2: str = "grokking_step") -> Dict[str, float]:
    """Compute correlations between two continuous variables."""
    valid_df = df.dropna(subset=[col1, col2])

    x = valid_df[col1].values
    y = valid_df[col2].values

    if len(x) < 2:
        return {"pearson": np.nan, "spearman": np.nan, "pearson_p": np.nan, "spearman_p": np.nan}

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p
    }

def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format a pandas DataFrame as a LaTeX table."""
    latex = df.to_latex(index=False, float_format="%.4f")

    # Wrap in table environment
    formatted = f"\\begin{{table}}[h]\n\\centering\n{latex}\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    return formatted

def main():
    try:
        df = load_results()

        # We need to analyze actual grokking step, treating missing grokking as max steps or NaN
        # For non-grokked models, we might set grokking step to max_steps for some analyses
        df_imputed = df.copy()
        mask_not_grokked = (df_imputed["grokked"] == False) & df_imputed["grokking_step"].notna()
        # Set to max steps if they didn't grok
        df_imputed.loc[mask_not_grokked, "grokking_step"] = df_imputed.loc[mask_not_grokked, "config_max_steps"]

        output_dir = Path("analysis")
        output_dir.mkdir(exist_ok=True)

        latex_output = []

        print("1. Computing Confidence Intervals (Final Test Accuracy)")
        ci_acc = compute_confidence_intervals(df, "final_test_acc")
        print(ci_acc)
        latex_output.append(format_latex_table(ci_acc, "95\\% Confidence Intervals for Final Test Accuracy", "tab:ci_acc"))

        print("\n2. Computing Confidence Intervals (Grokking Step)")
        # For grokking step, we only want to look at runs that actually grokked or we impute? Let's use imputed for now
        ci_grok = compute_confidence_intervals(df_imputed, "grokking_step")
        print(ci_grok)
        latex_output.append(format_latex_table(ci_grok, "95\\% Confidence Intervals for Grokking Step (Non-grokked imputed to max steps)", "tab:ci_grok"))

        print("\n3. Mann-Whitney U Tests vs Pure (Grokking Step)")
        mw_grok = mann_whitney_u_test(df_imputed, "grokking_step", baseline_group="pure")
        print(mw_grok)
        latex_output.append(format_latex_table(mw_grok, "Mann-Whitney U Test comparing Grokking Step against Pure baseline", "tab:mw_grok"))

        print("\n4. Correlations (Collapse Level vs Grokking Step)")
        # Filter to runs that actually have a collapse level defined and vary
        corr_df = df_imputed[df_imputed["config_collapse_level"].notna()]
        correlations = compute_correlations(corr_df, "config_collapse_level", "grokking_step")
        print(correlations)

        corr_df_summary = pd.DataFrame([correlations])
        latex_output.append(format_latex_table(corr_df_summary, "Correlation between Collapse Level and Grokking Step", "tab:corr_collapse_grok"))

        # Save LaTeX tables
        with open(output_dir / "statistical_summary.tex", "w") as f:
            f.write("% Statistical Analysis Summary\n\n")
            f.write("\n\n".join(latex_output))

        print(f"\nSaved LaTeX summary to {output_dir / 'statistical_summary.tex'}")

    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == "__main__":
    main()
