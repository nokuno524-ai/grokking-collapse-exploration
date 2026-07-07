import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

from analysis.statistical_tests import (
    compute_welch_ttest,
    compute_ks_test,
    compute_bootstrap_ci,
    compute_cohens_d
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
COLLAPSE_LEVELS = {
    "pure": 0.0,
    "low_collapse": 0.05,
    "medium_collapse": 0.15,
    "high_collapse": 0.30,
    "severe_collapse": 0.50
}

def parse_results(results_dir):
    """Parse results.json files from the reproduction directory."""
    data = []
    base_path = Path(results_dir)

    if not base_path.exists():
        print(f"Warning: {results_dir} does not exist.")
        return pd.DataFrame()

    for condition in SEVERITY_ORDER:
        condition_path = base_path / condition
        if not condition_path.exists():
            continue

        for seed_dir in condition_path.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            seed = int(seed_dir.name.split("_")[1])
            results_file = seed_dir / "results.json"

            if not results_file.exists():
                continue

            with open(results_file, 'r') as f:
                res = json.load(f)

            data.append({
                "condition": condition,
                "collapse_level": COLLAPSE_LEVELS[condition],
                "seed": seed,
                "final_test_acc": res.get("final_test_acc", 0.0),
                "final_weight_norm": res.get("final_weight_norm", 0.0),
                "grokking_step": res.get("grokking_step") if res.get("grokked") else np.nan,
                "grokked": res.get("grokked", False),
                "final_fourier_concentration": res.get("final_fourier_concentration", 0.0)
            })

    return pd.DataFrame(data)

def generate_summary_table(df, output_path):
    """Generate LaTeX summary table."""
    if df.empty:
        return

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline")
    lines.append("Condition & Final Test Acc & Weight Norm & Fourier Conc. & Grok Step (CI) & Grokked (\\%) \\\\")
    lines.append("\\hline")

    for condition in SEVERITY_ORDER:
        cond_df = df[df["condition"] == condition]
        if cond_df.empty:
            continue

        test_acc = f"{cond_df['final_test_acc'].mean():.3f} $\\pm$ {cond_df['final_test_acc'].std():.3f}"
        w_norm = f"{cond_df['final_weight_norm'].mean():.1f} $\\pm$ {cond_df['final_weight_norm'].std():.1f}"
        fourier = f"{cond_df['final_fourier_concentration'].mean():.3f} $\\pm$ {cond_df['final_fourier_concentration'].std():.3f}"

        grok_steps = cond_df["grokking_step"].dropna()
        if len(grok_steps) > 0:
            low, high = compute_bootstrap_ci(grok_steps)
            grok_step = f"{grok_steps.mean():.0f} [{low:.0f}, {high:.0f}]"
        else:
            grok_step = "N/A"

        grokked_pct = (cond_df["grokked"].sum() / len(cond_df)) * 100

        lines.append(f"{condition.replace('_', ' ').title()} & {test_acc} & {w_norm} & {fourier} & {grok_step} & {grokked_pct:.0f}\\% \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Summary of model collapse metrics across varying severity conditions.}")
    lines.append("\\label{tab:collapse_summary}")
    lines.append("\\end{table}")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {output_path}")

def run_statistical_tests(df, output_path):
    """Run tests comparing pure condition to severe_collapse."""
    if df.empty:
        return

    pure = df[df["condition"] == "pure"]
    severe = df[df["condition"] == "severe_collapse"]

    if pure.empty or severe.empty:
        print("Missing 'pure' or 'severe_collapse' data for statistical tests.")
        return

    with open(output_path, 'w') as f:
        f.write("# Statistical Analysis Results\n\n")

        # Test Accuracy (Welch's t-test)
        t_stat, p_val = compute_welch_ttest(pure["final_test_acc"], severe["final_test_acc"])
        d = compute_cohens_d(pure["final_test_acc"], severe["final_test_acc"])
        f.write(f"## Final Test Accuracy (Pure vs Severe Collapse)\n")
        f.write(f"- Welch's t-test: t={t_stat:.4f}, p={p_val:.4e}\n")
        f.write(f"- Cohen's d: {d:.4f}\n\n")

        # Weight Norm (KS test)
        ks_stat, p_val = compute_ks_test(pure["final_weight_norm"], severe["final_weight_norm"])
        f.write(f"## Weight Norm Distribution (Pure vs Severe Collapse)\n")
        f.write(f"- Kolmogorov-Smirnov test: KS={ks_stat:.4f}, p={p_val:.4e}\n\n")

    print(f"Statistical tests results saved to {output_path}")

def plot_metrics(df, output_dir):
    """Generate publication-quality figures."""
    if df.empty:
        return

    output_dir = Path(output_dir)

    # 1. Test Accuracy Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="condition", y="final_test_acc", data=df, order=SEVERITY_ORDER, palette="viridis")
    plt.axhline(y=0.95, color='r', linestyle='--', label='Grokking Threshold')
    plt.title("Final Test Accuracy by Collapse Severity")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Condition")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_accuracy_boxplot.png", dpi=300)
    plt.close()

    # 2. Weight Norm vs Fourier Concentration
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="final_weight_norm", y="final_fourier_concentration",
                    hue="condition", hue_order=SEVERITY_ORDER,
                    data=df, palette="viridis", s=100, alpha=0.8)
    plt.title("Fourier Concentration vs Weight Norm")
    plt.xlabel("Final Weight Norm")
    plt.ylabel("Fourier Concentration")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "weight_vs_fourier.png", dpi=300)
    plt.close()

    # 3. Correlation Matrix
    numeric_cols = ["collapse_level", "final_test_acc", "final_weight_norm", "final_fourier_concentration"]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Matrix of Metrics")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=300)
    plt.close()

    print(f"Plots saved to {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze reproduction results.")
    parser.add_argument("--results-dir", type=str, default="results/reproduce",
                        help="Directory containing reproduction results")
    parser.add_argument("--output-dir", type=str, default="analysis",
                        help="Directory to save analysis outputs")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = parse_results(args.results_dir)
    if df.empty:
        print("No data found to analyze.")
        return

    df.to_csv(out_path / "aggregated_results.csv", index=False)

    generate_summary_table(df, out_path / "summary_table.tex")
    run_statistical_tests(df, out_path / "statistical_tests.md")
    plot_metrics(df, args.output_dir)

if __name__ == "__main__":
    main()
