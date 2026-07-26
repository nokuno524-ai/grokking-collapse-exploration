import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from tabulate import tabulate

def get_grokking_step(history, threshold=0.9):
    for h in history:
        if h.get("test_acc", 0.0) >= threshold:
            return h["step"]
    return float('inf')

def load_all_metrics():
    data = []
    base_dir = "results"

    grid_dir = os.path.join(base_dir, "grid")

    if os.path.exists(grid_dir):
        for root, dirs, files in os.walk(grid_dir):
            if "results.json" in files:
                with open(os.path.join(root, "results.json"), "r") as f:
                    res = json.load(f)
                    config = res.get("config", {})
                    history = res.get("history", [])

                    collapse_level = config.get("collapse_level", 0.0)
                    severity = config.get("collapse_severity", 0.5)
                    seed = config.get("seed", 42)
                    d_model = config.get("d_model", 128)

                    grok_step = get_grokking_step(history)

                    data.append({
                        "level": collapse_level,
                        "severity": severity,
                        "seed": seed,
                        "d_model": d_model,
                        "grok_step": grok_step,
                        "grokked": 1 if grok_step != float('inf') else 0
                    })

    for entry in os.scandir(base_dir):
        if entry.is_dir() and entry.name not in ["grid", "attention", "circuits", "dashboard", "statistics"]:
            res_file = os.path.join(entry.path, "results.json")
            if os.path.exists(res_file):
                with open(res_file, "r") as f:
                    res = json.load(f)
                    config = res.get("config", {})
                    history = res.get("history", [])

                    collapse_level = config.get("collapse_level", 0.0)
                    severity = config.get("collapse_severity", 0.5)
                    seed = config.get("seed", 42)
                    d_model = config.get("d_model", 128)

                    grok_step = get_grokking_step(history)

                    if entry.name == "pure": collapse_level = 0.0
                    elif entry.name == "low_collapse": collapse_level = 0.25
                    elif entry.name == "medium_collapse": collapse_level = 0.5
                    elif entry.name == "severe_collapse": collapse_level = 0.75
                    elif entry.name == "high_collapse": collapse_level = 1.0

                    data.append({
                        "level": collapse_level,
                        "severity": severity,
                        "seed": seed,
                        "d_model": d_model,
                        "grok_step": grok_step,
                        "grokked": 1 if grok_step != float('inf') else 0
                    })

    return pd.DataFrame(data)

def bootstrap_ci(data, statistic_func, n_bootstraps=1000, ci=95):
    data = data[~np.isnan(data) & (data != float('inf'))]
    if len(data) == 0:
        return np.nan, np.nan

    bootstrapped_stats = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_stats.append(statistic_func(sample))

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    return np.percentile(bootstrapped_stats, lower_percentile), np.percentile(bootstrapped_stats, upper_percentile)

def run_statistical_tests(df, output_path):
    with open(output_path, "w") as f:
        f.write("# Statistical Analysis Results\n\n")

        pure_grok_steps = df[df["level"] == 0.0]["grok_step"].replace(float('inf'), np.nan).dropna().values
        if len(pure_grok_steps) > 0:
            ci_lower, ci_upper = bootstrap_ci(pure_grok_steps, np.mean)
            mean_grok = np.mean(pure_grok_steps)
            f.write("## 1. Bootstrap Confidence Intervals\n")
            f.write(f"- Pure model mean grokking step: {mean_grok:.1f} (95% CI: [{ci_lower:.1f}, {ci_upper:.1f}])\n\n")

        f.write("## 2. Hypothesis Testing\n")
        low_col_steps = df[df["level"] == 0.25]["grok_step"].replace(float('inf'), np.nan).dropna().values

        if len(pure_grok_steps) > 1 and len(low_col_steps) > 1:
            t_stat, p_val = stats.ttest_ind(pure_grok_steps, low_col_steps, equal_var=False)
            f.write(f"- Welch's t-test comparing Pure and Low Collapse grokking steps:\n")
            f.write(f"  - t-statistic: {t_stat:.4f}\n")
            f.write(f"  - p-value: {p_val:.4e}\n")
            if p_val < 0.05:
                f.write("  - Result: Collapse significantly delays grokking (p < 0.05).\n\n")
            else:
                f.write("  - Result: No significant difference detected.\n\n")
        else:
             f.write("  - Not enough data to perform t-test across multiple seeds.\n\n")

        f.write("## 3. Correlation Analysis\n")
        corr, p_val = stats.spearmanr(df["level"], df["grok_step"])
        f.write(f"- Spearman correlation between collapse level and grokking step:\n")
        f.write(f"  - correlation: {corr:.4f}\n")
        f.write(f"  - p-value: {p_val:.4e}\n\n")

        f.write("## 4. Multiple Regression: Predicting Grokking Step\n")
        # To perform regression on step, we cap float('inf') to 50000 (max steps)
        # so that OLS can process it representing "failed to grok in given time".
        df_reg = df.copy()
        df_reg["grok_step_capped"] = df_reg["grok_step"].replace(float('inf'), 50000)

        X = sm.add_constant(df_reg[["level", "severity", "d_model"]])
        y = df_reg["grok_step_capped"]

        try:
            model = sm.OLS(y, X)
            result = model.fit()
            f.write("### OLS Regression Results\n")

            latex_table = result.summary().as_latex()
            f.write("```latex\n")
            f.write(latex_table)
            f.write("```\n\n")
        except Exception as e:
            f.write(f"Regression failed to fit: {str(e)}\n\n")

def main():
    os.makedirs("results/statistics", exist_ok=True)
    df = load_all_metrics()

    if len(df) == 0:
        print("No metrics found to run statistics.")
        return

    run_statistical_tests(df, "results/statistics/statistical_report.md")
    print("Statistical report generated at results/statistics/statistical_report.md")

if __name__ == "__main__":
    main()
