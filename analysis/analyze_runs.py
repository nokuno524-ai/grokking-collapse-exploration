"""
Analysis script to process results/summary.csv and compute statistics
on grokking steps, weight norms, and output comprehensive reports.
"""
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

def compute_bootstrap_ci(data, statistic_fn, num_bootstraps=1000, ci=95):
    """Compute bootstrap confidence interval for a given statistic."""
    data = np.array(data)
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    bootstraps = np.random.choice(data, size=(num_bootstraps, n), replace=True)
    stats = np.apply_along_axis(statistic_fn, 1, bootstraps)
    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return np.mean(stats), lower, upper

def main():
    rows = []
    if not Path("results/summary.csv").exists():
        print("results/summary.csv not found. Please run scripts/inventory.py first.")
        return

    with open("results/summary.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group grokking steps by collapse ratio
    grokking_steps_by_ratio = defaultdict(list)
    wn_grokked = []
    wn_not_grokked = []
    wn_by_ratio = defaultdict(list)

    for row in rows:
        ratio = float(row["collapse_ratio"])
        wn = float(row["final_weight_norm"])

        wn_by_ratio[ratio].append(wn)

        if row["grokking_step"]:
            step = float(row["grokking_step"])
            grokking_steps_by_ratio[ratio].append(step)
            wn_grokked.append(wn)
        else:
            wn_not_grokked.append(wn)

    results = {}
    grokking_curve = []

    for ratio, steps in grokking_steps_by_ratio.items():
        if not steps:
            continue
        mean_val, lower, upper = compute_bootstrap_ci(steps, np.mean)
        grokking_curve.append({
            "collapse_ratio": ratio,
            "mean_grokking_step": float(mean_val),
            "lower_ci": float(lower),
            "upper_ci": float(upper),
            "count": len(steps)
        })

    results["grokking_curve"] = sorted(grokking_curve, key=lambda x: x["collapse_ratio"])

    mean_wn_grokked = np.mean(wn_grokked) if wn_grokked else 0.0
    mean_wn_not_grokked = np.mean(wn_not_grokked) if wn_not_grokked else 0.0

    results["weight_norm_analysis"] = {
        "mean_final_weight_norm_grokked": float(mean_wn_grokked),
        "mean_final_weight_norm_not_grokked": float(mean_wn_not_grokked),
        "grokking_success_rate": float(len(wn_grokked) / len(rows))
    }

    results["weight_norm_by_collapse_ratio"] = sorted([
        {"collapse_ratio": ratio, "mean_final_weight_norm": float(np.mean(wns))}
        for ratio, wns in wn_by_ratio.items()
    ], key=lambda x: x["collapse_ratio"])

    with open("results/report.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    markdown_lines = []
    markdown_lines.append("# Analysis Report: Grokking and Model Collapse\n")
    markdown_lines.append("## Grokking Step vs Collapse Severity (Ratio)\n")
    markdown_lines.append("| Collapse Ratio | Mean Grokking Step | 95% CI Lower | 95% CI Upper | Count |")
    markdown_lines.append("| --- | --- | --- | --- | --- |")
    for r in results["grokking_curve"]:
        markdown_lines.append(f"| {r['collapse_ratio']} | {r['mean_grokking_step']:.1f} | {r['lower_ci']:.1f} | {r['upper_ci']:.1f} | {r['count']} |")

    markdown_lines.append("\n## Weight Norm Analysis\n")
    markdown_lines.append(f"- **Mean final weight norm for runs that grokked:** {mean_wn_grokked:.2f}")
    markdown_lines.append(f"- **Mean final weight norm for runs that failed to grok:** {mean_wn_not_grokked:.2f}")
    markdown_lines.append("\nThis suggests that higher final weight norms are associated with the failure to grok, indicating that weight norm may mediate or serve as a proxy for grokking failure under collapse.")

    markdown_lines.append("\n## Average Final Weight Norm by Collapse Ratio\n")
    markdown_lines.append("| Collapse Ratio | Mean Final Weight Norm |")
    markdown_lines.append("| --- | --- |")
    for r in results["weight_norm_by_collapse_ratio"]:
        markdown_lines.append(f"| {r['collapse_ratio']} | {r['mean_final_weight_norm']:.2f} |")

    with open("results/REPORT.md", "w") as f:
        f.write("\n".join(markdown_lines))

    print("Report generated at results/REPORT.md and results/report.json")

if __name__ == "__main__":
    main()
